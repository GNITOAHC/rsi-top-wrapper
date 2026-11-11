#!/usr/bin/env python3
"""Meta-Harness search on the Anthropic performance takehome.

Native search: four concurrent proposer sessions (Pi + DeepSeek). Each session
reads the whole search archive (every valid program, every attempt's evidence,
the evolution summary, and the frontier) and writes one new candidate through
a confined write tool; the loop evaluates it with the shared two-phase
evaluator and refills the slot. Proposers never run benchmarks.

The attempts trace uses the OpenEvolve event schema, so the shared dashboard,
planner, fork, and export tooling read it unchanged. An optional breadth
manager (``mh_breadth``) adds RSI-top interventions around this loop.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OE = ROOT / "integrations" / "openevolve-takehome"
for import_root in (HERE, OE, OE / "vendor" / "openevolve", ROOT / "shared" / "meta-agent-audit"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
# pi_cli reads its provider defaults at import time.
os.environ.setdefault("AUDIT_PI_PROVIDER", "deepseek")
os.environ.setdefault("AUDIT_PI_MODEL", "deepseek-v4-flash")
os.environ.setdefault("AUDIT_PI_API_KEY_ENV", "DEEPSEEK_API_KEY")

import evaluator as oe_evaluator  # noqa: E402
from deepseek_llm import DeepSeekCostLedger  # noqa: E402
from meta_audit.source_snapshot import capture_sources, write_source_snapshot  # noqa: E402
from pi_cli import build_pi_argv, build_pi_subprocess_env, parse_pi_stream  # noqa: E402

WORKER_PROMPT = ROOT / "prompts" / "worker_metaharness.md"
ROOT_PROGRAM = OE / "initial_program.py"
TAKEHOME = OE / "vendor" / "takehome"
PI_TOOLS = "read_public_file,list_public_directory,write_candidate"
PI_EXTENSIONS = (OE / "pi_public_files_extension.ts", HERE / "pi_candidate_write_extension.ts")
NATIVE_CONCURRENCY = 4  # matches OpenEvolve/Shinka ordinary concurrency
PI_TIMEOUT_SECONDS = 3600.0  # one multi-turn session; OpenEvolve allows 2400 s per single call
MAX_CODE_CHARS = 60000  # OpenEvolve max_code_length
ROOT_ID = "root"
ORDINARY_MESSAGE = (
    "Run one Meta-Harness proposal step: read the archive in the working "
    "directory, then call write_candidate exactly once with one new program."
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Trace:
    """Append-only attempts trace with the OpenEvolve event envelope."""

    def __init__(self, path: Path, run_instance_id: str) -> None:
        self.path = path
        self.run_instance_id = run_instance_id

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "schema_version": 2,
            "event_id": str(uuid.uuid4()),
            "event": event,
            "timestamp": time.time(),
            "run_instance_id": self.run_instance_id,
            "pid": os.getpid(),
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        return record


def read_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def evaluate_program(path: Path) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    result = oe_evaluator.evaluate(str(path))
    metrics = {key: float(value) for key, value in result.metrics.items()}
    artifacts = {key: str(value)[-4000:] for key, value in (result.artifacts or {}).items()}
    return "failure_stage" not in artifacts, metrics, artifacts


class Search:
    def __init__(
        self,
        *,
        output: Path,
        budget: float,
        worker_model: str,
        worker_reasoning: str,
        pi: str,
        wall_time_seconds: float,
        seed: int,
        breadth: Any = None,
    ) -> None:
        self.output = output
        self.budget = float(budget)
        self.worker_model = worker_model
        self.worker_reasoning = worker_reasoning
        self.pi = pi
        self.seed = seed
        self.deadline = time.monotonic() + float(wall_time_seconds)
        self.breadth = breadth
        self.archive = output / "archive"
        self.trace = Trace(output / "attempts.jsonl", str(uuid.uuid4()))
        self.ledger = DeepSeekCostLedger(str(output / "cost_ledger.json"), model=worker_model)
        self.pool = ThreadPoolExecutor(max_workers=48)
        self.programs: dict[str, dict[str, Any]] = {}
        self.summary: list[dict[str, Any]] = []
        self.best_id = ROOT_ID
        self.next_seq = 0
        self.next_iteration = 0
        self._stop = asyncio.Event()
        self._running: dict[asyncio.Task, dict[str, Any]] = {}

    # ---------- state and archive ----------

    def _cycles(self, program_id: str) -> float:
        return float(self.programs[program_id]["metrics"]["cycles"])

    def _set_root(self, code: str, metrics: dict[str, Any]) -> None:
        self.programs[ROOT_ID] = {
            "id": ROOT_ID, "code": code, "parent_id": None, "generation": 0,
            "metrics": metrics, "iteration": None, "changes": "Starting program.",
        }

    def replay(self, events: list[dict[str, Any]]) -> None:
        """Rebuild search state from a copied trace prefix (fork)."""
        started = next(row for row in events if row.get("event") == "run_started")
        self._set_root(started["root_code"], started["root_metrics"])
        completions = {}
        for row in events:
            if row.get("event") == "canonical_completion":
                completions[int(row["completion_seq"])] = row
                self.next_seq = max(self.next_seq, int(row["completion_seq"]) + 1)
                self.next_iteration = max(self.next_iteration, int(row["iteration"]) + 1)
        for seq in sorted(completions):
            row = completions[seq]
            child = row.get("child") or {}
            if row.get("evaluation_valid") is True and child.get("id"):
                self._add_program(child, iteration=int(row["iteration"]))
            self._record_summary(row)

    def _add_program(self, child: dict[str, Any], *, iteration: int) -> None:
        self.programs[child["id"]] = {
            "id": child["id"], "code": child["code"], "parent_id": child.get("parent_id"),
            "generation": child.get("generation"), "metrics": child["metrics"],
            "iteration": iteration, "changes": child.get("changes_description"),
        }
        if self._cycles(child["id"]) < self._cycles(self.best_id):
            self.best_id = child["id"]
        (self.archive / "candidates").mkdir(parents=True, exist_ok=True)
        (self.archive / "candidates" / f"{child['id']}.py").write_text(child["code"], encoding="utf-8")

    def _record_summary(self, completion: dict[str, Any]) -> None:
        seq = int(completion["completion_seq"])
        child = completion.get("child") or {}
        cycles = (child.get("metrics") or {}).get("cycles") if completion.get("evaluation_valid") else None
        row = {
            "seq": seq,
            "attempt_id": completion["attempt_id"],
            "program_id": child.get("id") if completion.get("evaluation_valid") else None,
            "base_program_id": completion.get("parent_id"),
            "valid": bool(completion.get("evaluation_valid")),
            "cycles": cycles,
            "failure_stage": completion.get("failure_stage"),
            "guided_intervention": bool(completion.get("meta_intervention")),
            "changes": (completion.get("metadata") or {}).get("changes"),
        }
        self.summary.append(row)
        evidence = self.archive / "attempts" / f"{seq:08d}"
        evidence.mkdir(parents=True, exist_ok=True)
        if isinstance(child.get("code"), str):
            (evidence / "code.py").write_text(child["code"], encoding="utf-8")
        write_json(evidence / "result.json", {
            **row, "metrics": child.get("metrics"), "artifacts": completion.get("artifacts"),
        })
        with (self.archive / "evolution_summary.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._write_frontier()

    def _write_frontier(self) -> None:
        ranked = sorted(self.programs.values(), key=lambda item: float(item["metrics"]["cycles"]))
        write_json(self.archive / "frontier.json", {
            "best_program_id": self.best_id,
            "best_cycles": self._cycles(self.best_id),
            "valid_program_count": len(self.programs),
            "completed_attempt_count": len(self.summary),
            "top_programs": [
                {"id": item["id"], "cycles": item["metrics"]["cycles"], "parent_id": item["parent_id"],
                 "changes": item["changes"], "code_path": f"candidates/{item['id']}.py"}
                for item in ranked[:10]
            ],
        })

    def _write_best(self) -> None:
        best = self.programs[self.best_id]
        target = self.output / "best"
        target.mkdir(parents=True, exist_ok=True)
        (target / "best_program.py").write_text(best["code"], encoding="utf-8")
        write_json(target / "best_program_info.json", {
            "id": best["id"], "generation": best["generation"], "metrics": best["metrics"],
            "parent_id": best["parent_id"], "changes": best["changes"],
        })

    def setup(self, prefix: Path | None) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        reference = self.archive / "reference"
        reference.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TAKEHOME / "problem.py", reference)
        shutil.copy2(OE / "evaluator.py", reference)
        (self.output / "system_prompt.txt").write_text(
            WORKER_PROMPT.read_text(encoding="utf-8")
            + "\n\n```python\n" + (TAKEHOME / "problem.py").read_text(encoding="utf-8") + "\n```\n",
            encoding="utf-8",
        )
        sources = capture_sources(ROOT)
        write_json(self.output / "run_manifest.json", {
            "schema": "mh_run_manifest_v1",
            "argv": sys.argv,
            "created_at": time.time(),
            "budget_usd": self.budget,
            "worker": {"provider": "deepseek", "model": self.worker_model, "reasoning": self.worker_reasoning},
            "native_concurrency": NATIVE_CONCURRENCY,
            "random_seed": self.seed,
            "branch_prefix": str(prefix) if prefix else None,
            "source_provenance": write_source_snapshot(self.output, sources),
        })
        if prefix is not None:
            data = prefix.read_bytes()
            (self.output / "attempts.jsonl").write_bytes(data)
            self.replay(read_events(self.output / "attempts.jsonl"))
        else:
            code = ROOT_PROGRAM.read_text(encoding="utf-8")
            valid, metrics, artifacts = evaluate_program(ROOT_PROGRAM)
            if not valid:
                raise RuntimeError(f"starting program failed evaluation: {artifacts}")
            self._set_root(code, metrics)
        (self.archive / "candidates").mkdir(parents=True, exist_ok=True)
        (self.archive / "candidates" / f"{ROOT_ID}.py").write_text(self.programs[ROOT_ID]["code"], encoding="utf-8")
        self._write_frontier()
        self.trace.emit(
            "run_started",
            budget_usd=self.budget,
            random_seed=self.seed,
            worker_model=self.worker_model,
            worker_reasoning=self.worker_reasoning,
            native_concurrency=NATIVE_CONCURRENCY,
            output_dir=str(self.output),
            root_program_id=ROOT_ID,
            root_code=self.programs[ROOT_ID]["code"],
            root_metrics=self.programs[ROOT_ID]["metrics"],
            initial_program_sha256=sha256_text(self.programs[ROOT_ID]["code"]),
            branch_prefix=str(prefix) if prefix else None,
        )
        self._write_best()

    # ---------- one attempt ----------

    async def _run_pi(self, workspace: Path, message: str) -> dict[str, Any]:
        view = workspace / "view"
        home, tmp = workspace / "pi-home", workspace / "pi-tmp"
        home.mkdir()
        tmp.mkdir()
        argv = build_pi_argv(
            self.worker_model, str(self.output / "system_prompt.txt"), self.pi,
            extra_extensions=PI_EXTENSIONS, tools=PI_TOOLS, user_message=message,
            provider="deepseek", reasoning=self.worker_reasoning,
        )
        env = build_pi_subprocess_env(self.pi, home=home, tmpdir=tmp, provider="deepseek",
                                      api_key_env="DEEPSEEK_API_KEY")
        stdout_path = workspace / "pi.jsonl"
        with stdout_path.open("w") as stdout, (workspace / "pi.stderr.txt").open("w") as stderr:
            process = await asyncio.create_subprocess_exec(
                *argv, cwd=view, env=env, stdin=asyncio.subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=PI_TIMEOUT_SECONDS)
            finally:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()
        stream = parse_pi_stream(stdout_path.read_text(encoding="utf-8", errors="replace"))
        stream["returncode"] = process.returncode
        return stream

    def _message(self, guidance: dict[str, Any] | None) -> str:
        if guidance is None:
            return ORDINARY_MESSAGE
        return self.breadth.render_guidance(guidance)

    async def _attempt(self, iteration: int, guidance: dict[str, Any] | None) -> None:
        attempt_id = f"{self.trace.run_instance_id}:{iteration}"
        scheduled_at = time.time()
        if self.breadth is not None:
            self.breadth.on_scheduled(attempt_id, iteration, scheduled_at, guidance)
        workspace = self.output / "workspaces" / f"{iteration:05d}"
        workspace.mkdir(parents=True)
        shutil.copytree(self.archive, workspace / "view")
        (workspace / "view" / "out").mkdir()
        stream: dict[str, Any] = {}
        terminal, failure, code, pending = "worker_error", None, None, {}
        try:
            try:
                stream = await self._run_pi(workspace, self._message(guidance))
            except asyncio.TimeoutError:
                failure = "proposal_timeout"
            out = workspace / "view" / "out"
            if failure is None and (out / "candidate.py").is_file() and (out / "pending.json").is_file():
                code = (out / "candidate.py").read_text(encoding="utf-8")
                pending = json.loads((out / "pending.json").read_text(encoding="utf-8"))
            elif failure is None:
                failure = "proposal_failed"
        except asyncio.CancelledError:
            failure, terminal = "abandoned", "abandoned_at_shutdown"
            if (workspace / "pi.jsonl").is_file():  # charge the partial session
                stream = parse_pi_stream((workspace / "pi.jsonl").read_text(encoding="utf-8", errors="replace"))
        except Exception:
            failure = "proposal_failed"
            (workspace / "loop_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        usage = {
            "input_tokens": int(stream.get("input_tokens") or 0)
            + int(stream.get("cache_read_tokens") or 0) + int(stream.get("cache_write_tokens") or 0),
            "output_tokens": int(stream.get("output_tokens") or 0),
        }
        cost = self.ledger.price(usage)
        self.ledger.add(usage)

        declared = pending.get("base_id")
        if guidance is not None:
            parent_id = guidance["target_branch_id"]
        elif declared in self.programs:
            parent_id = declared
        elif code is not None:
            parent_id = self.best_id  # ponytail: undeclared/unknown base attributed to the frontier
        else:
            parent_id = None
        parent = self.programs.get(parent_id) if parent_id else None
        read_ids = sorted({
            Path(path).stem for path in stream.get("successful_read_paths") or []
            if str(path).startswith("candidates/") and Path(path).stem in self.programs
        } - {parent_id})
        self.trace.emit(
            "attempt_scheduled",
            timestamp=scheduled_at,  # proposal start; emitted once the base is known
            emitted_at=time.time(),
            attempt_id=attempt_id,
            iteration=iteration,
            target_island=0,
            parent_id=parent_id,
            parent_code_sha256=sha256_text(parent["code"]) if parent else None,
            parent_metrics=parent["metrics"] if parent else None,
            inspiration_ids=read_ids,
            meta_intervention=guidance is not None,
            meta_audit_id=guidance.get("audit_id") if guidance else None,
        )

        valid, metrics, artifacts = False, {}, {}
        if code is not None and failure is None:
            if len(code) > MAX_CODE_CHARS:
                failure, terminal = "code_too_long", "evaluation_failed"
            else:
                candidate = workspace / "candidate.py"
                candidate.write_text(code, encoding="utf-8")
                try:
                    valid, metrics, artifacts = await asyncio.get_running_loop().run_in_executor(
                        self.pool, evaluate_program, candidate
                    )
                except asyncio.CancelledError:
                    failure, terminal = "abandoned", "abandoned_at_shutdown"
                except Exception:
                    failure, terminal = "evaluation_error", "evaluation_failed"
                    (workspace / "loop_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                if failure is None:
                    terminal = "committed" if valid else "evaluation_failed"
                    failure = None if valid else artifacts.get("failure_stage")
        # The archive snapshot is reproducible from the trace; keep only this session's evidence.
        if (workspace / "view" / "out").is_dir():
            shutil.move(str(workspace / "view" / "out"), str(workspace / "out"))
        shutil.rmtree(workspace / "view", ignore_errors=True)
        self._complete(
            attempt_id=attempt_id, iteration=iteration, scheduled_at=scheduled_at, guidance=guidance,
            parent_id=parent_id, parent=parent, code=code, pending=pending, valid=valid, metrics=metrics,
            artifacts=artifacts, failure=failure, terminal=terminal, cost=cost, usage=usage,
            stream=stream, read_ids=read_ids,
        )

    def _complete(self, *, attempt_id, iteration, scheduled_at, guidance, parent_id, parent, code, pending,
                  valid, metrics, artifacts, failure, terminal, cost, usage, stream, read_ids) -> None:
        seq = self.next_seq
        self.next_seq += 1
        now = time.time()
        program_id = f"p{iteration:05d}"
        changes = str(pending.get("changes") or "No recorded change summary.")
        child = None
        if code is not None:
            child = {
                "id": program_id, "code": code, "code_sha256": sha256_text(code),
                "parent_id": parent_id,
                "generation": (int(parent["generation"]) + 1) if parent else None,
                "metrics": metrics, "changes_description": changes,
                "metadata": {"changes": changes, "declared_base_id": pending.get("base_id")},
            }
        completion = self.trace.emit(
            "canonical_completion",
            completion_seq=seq, commit_index=seq, attempt_id=attempt_id, iteration=iteration,
            status="completed" if terminal == "committed" else "failed", terminal=terminal,
            scheduled_at=scheduled_at, seconds_since_scheduled=now - scheduled_at,
            scheduled_parent_id=parent_id, scheduled_target_island=0, parent_id=parent_id,
            target_island=0, evaluation_valid=valid, failure_stage=None if valid else (failure or "invalid"),
            child=child, artifacts=artifacts, metadata={"changes": changes},
            api_cost_usd=cost, usage=usage, inspiration_ids=read_ids,
            pi={"returncode": stream.get("returncode"), "tool_calls": stream.get("tool_calls"),
                "read_count": len(stream.get("successful_read_paths") or [])},
            meta_intervention=guidance is not None,
            meta_audit_id=guidance.get("audit_id") if guidance else None,
            direction_index=guidance.get("direction_index") if guidance else None,
        )
        if valid:
            self._add_program(child, iteration=iteration)
            self.trace.emit(
                "program_committed",
                completion_seq=seq, commit_index=seq, attempt_id=attempt_id, program_id=program_id,
                parent_id=parent_id, generation=child["generation"], iteration_found=iteration,
                metrics=metrics, metadata=child["metadata"], code_sha256=child["code_sha256"],
                database_program_count=len(self.programs),
            )
            self.trace.emit(
                "db_state",
                completion_seq=seq, program_id=program_id, db_state_schema="mh_db_state_v1",
                state={"best_program_id": self.best_id, "program_ids": sorted(self.programs)},
            )
            if self.best_id == program_id:
                self._write_best()
        self._record_summary(completion)
        if self.breadth is not None:
            self.breadth.on_completion(
                seq=seq, attempt_id=attempt_id, at=completion["timestamp"], valid=valid,
                cycles=metrics.get("cycles") if valid else None, child_id=program_id if valid else None,
                guidance=guidance,
            )

    # ---------- scheduler ----------

    def _launch(self, guidance: dict[str, Any] | None) -> None:
        iteration = self.next_iteration
        self.next_iteration += 1
        task = asyncio.create_task(self._attempt(iteration, guidance))
        self._running[task] = {"guided": guidance is not None}

    def _ordinary_running(self) -> int:
        return sum(not meta["guided"] for meta in self._running.values())

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, self._stop.set)
        if self.breadth is not None:
            self.breadth.bind(self)
        while not self._stop.is_set():
            if self.breadth is not None:
                self.breadth.poll()
            over_budget = self.ledger.total() >= self.budget
            past_deadline = time.monotonic() >= self.deadline
            if not over_budget:
                while self.breadth is not None and (guidance := self.breadth.take_guidance()) is not None:
                    self._launch(guidance)
                while (not past_deadline and self._ordinary_running() < NATIVE_CONCURRENCY
                       and (self.breadth is None or self.breadth.ordinary_allowed(len(self._running)))):
                    self._launch(None)
            # RSI-top finishes a triggered planner/cohort after the deadline, like OE/Shinka;
            # ordinary attempts still in flight at exit are recorded as abandoned.
            breadth_busy = self.breadth is not None and self.breadth.busy()
            if over_budget or (past_deadline and not breadth_busy):
                break
            if not self._running:
                await asyncio.sleep(1.0)
                continue
            done, _ = await asyncio.wait(self._running, timeout=2.0, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                self._running.pop(task)
                task.result()
        for task in self._running:
            task.cancel()
        if self._running:
            await asyncio.gather(*self._running, return_exceptions=True)
        if self.breadth is not None:
            self.breadth.shutdown()
        self.trace.emit(
            "run_finished", total_api_cost_usd=self.ledger.total(),
            completed_attempts=len(self.summary), best_program_id=self.best_id,
            best_cycles=self._cycles(self.best_id),
        )
        self._write_best()
        self.pool.shutdown(wait=False, cancel_futures=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=400.0, help="worker API budget estimate in USD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-model", default="deepseek-v4-flash")
    parser.add_argument("--worker-reasoning", default="high")
    parser.add_argument("--wall-time-seconds", type=float, default=12 * 3600.0)
    parser.add_argument("--prefix", type=Path, help="copied historical trace prefix to resume a fork from")
    parser.add_argument("--pi", default=os.environ.get("AUDIT_PI_CLI", "pi"))
    return parser.parse_args(argv)


def check_environment(pi: str) -> str:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    resolved = shutil.which(pi)
    if resolved is None:
        raise SystemExit(f"Pi runtime not found: {pi}; run ./setup.sh")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pi = check_environment(args.pi)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output already exists: {output}")
    search = Search(
        output=output, budget=args.budget, worker_model=args.worker_model,
        worker_reasoning=args.worker_reasoning, pi=pi, wall_time_seconds=args.wall_time_seconds,
        seed=args.seed,
    )
    search.setup(args.prefix)
    asyncio.run(search.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
