"""Repeated RSI-top breadth interventions around the Meta-Harness loop.

Same treatment as the OpenEvolve and Shinka integrations: after 20 active flat
minutes and 12 terminal ordinary attempts (and the minimum run age), freeze the
archive, build the shared planner view, run the shared 4x8 planner + dedup, and
dispatch every retained direction as its own guided proposer session beside
the in-flight work. Ordinary search resumes once fewer than the native number
of sessions remain. Trigger, journal kinds, and fields follow
``shinka-takehome/shinka_breadth.py``; the planner side is reused unchanged
through subprocesses.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OE = ROOT / "integrations" / "openevolve-takehome"
SH = ROOT / "integrations" / "shinka-takehome"
PLANNER_PROMPT = ROOT / "prompts" / "planner.md"
MEANINGFUL_RELATIVE_GAIN = 0.002
GUIDANCE_VERSION = "branch_recipe_portfolio_guidance_v2"
NATIVE_CONCURRENCY = 4


def load_oe_supervisor():
    """OpenEvolve supervisor helpers (cutoff, active time axis) under a unique name."""
    name = "oe_repeat_flat_breadth"
    if name not in sys.modules:
        if str(OE) not in sys.path:
            sys.path.insert(0, str(OE))
        spec = importlib.util.spec_from_file_location(name, OE / "run_repeat_flat_breadth.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(document: Any) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def render_guidance(payload: dict[str, Any]) -> str:
    donors = payload.get("donor_branch_ids") or []
    donor_text = "; ".join(f"{item} (candidates/{item}.py)" for item in donors) if donors else "(none)"
    constraints = "\n".join(f"- {condition}" for condition in payload["compatibility"])
    diagnosis = payload.get("diagnosis")
    diagnosis_text = (
        f"Independent planner diagnoses — context only, not separate tasks:\n{diagnosis}\n\n"
        if diagnosis else ""
    )
    return (
        "==== META-INTERVENTION DIAGNOSIS COHORT TASK "
        f"(audit {payload['audit_id']}, {payload['plan_version']}) ====\n"
        "Execute one selected intervention direction for this work unit.\n\n"
        f"{diagnosis_text}"
        f"Target program: {payload['target_branch_id']} "
        f"(candidates/{payload['target_branch_id']}.py) is the program you edit\n"
        f"Donor inspirations: {donor_text}\n"
        "Hypothesis:\n"
        f"{payload['diagnosed_bottleneck']}\n"
        "Action:\n"
        f"{payload['lane_task']}\n"
        "Constraints:\n"
        f"{constraints}\n\n"
        "Produce one concrete, bounded candidate for this action: read the target and "
        "any donors, then call write_candidate exactly once with base_id "
        f"{payload['target_branch_id']}.\n"
        "==== END META-INTERVENTION DIAGNOSIS COHORT TASK ===="
    )


def failed_directions(
    *, existing: list[dict[str, Any]], planner: Path, journal: Path, cohort: int
) -> list[dict[str, Any]]:
    """Port of the Shinka supervisor's prior-failed-intervention memory."""
    rows = [row for row in _rows(journal) if row.get("cohort_index") == cohort]
    if any(row.get("kind") == "meaningful_frontier_improvement" for row in rows):
        return []
    if not any(row.get("kind") == "guided_cohort_complete" for row in rows):
        return list(existing)
    directions = json.loads((planner / "portfolio.json").read_text(encoding="utf-8"))["directions"]
    assignments = {int(row["generation"]): int(row["direction_index"]) - 1
                   for row in rows if row.get("kind") == "direction_assigned"}
    terminals = {int(row["generation"]): row for row in rows if row.get("kind") == "guided_attempt_terminal"}
    frontier = next((row for row in rows if row.get("kind") == "cohort_bound"), {}).get("frontier_cycles")
    result = list(existing)
    for generation, index in sorted(assignments.items(), key=lambda item: item[1]):
        direction = directions[index]
        terminal = terminals.get(generation, {})
        cycles = terminal.get("cycles")
        if terminal.get("evaluation_valid") is not True or not isinstance(cycles, (int, float)):
            outcome, delta = "invalid", None
        elif not isinstance(frontier, (int, float)):
            outcome, delta = "unknown", None
        else:
            delta = float(frontier) - float(cycles)
            outcome = "tie" if delta >= 0 else "regress"
        result.append({
            "type": direction.get("type"),
            "mechanism_hypothesis": direction.get("mechanism_hypothesis"),
            "task": direction.get("task"),
            "target_branch_id": direction.get("target_branch_id"),
            "outcome": outcome,
            "frontier_delta_cycles": delta,
            "completion_seq": terminal.get("completion_seq"),
        })
    return result


class BreadthManager:
    def __init__(
        self,
        *,
        output: Path,
        time_axis_path: Path,
        epochs_dir: Path,
        flat_seconds: float = 1200.0,
        flat_min_terminal_attempts: int = 12,
        minimum_run_age_seconds: float = 0.0,
        start_ordinary: bool = False,
        planner_model: str = "deepseek-v4-flash",
        planner_provider: str = "deepseek",
        planner_reasoning: str = "high",
        planner_base_url: str = "https://api.deepseek.com",
        planner_api_key_env: str = "DEEPSEEK_API_KEY",
        metric_subspace_seed: int = 20260830,
    ) -> None:
        if flat_seconds <= 0 or flat_min_terminal_attempts < 1:
            raise ValueError("breadth trigger limits must be positive")
        self.output = Path(output).resolve()
        self.control_dir = self.output / "breadth_control"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.epochs_dir = Path(epochs_dir).resolve()
        self.time_axis_path = Path(time_axis_path).resolve()
        self.journal_path = self.output / "breadth_interventions.jsonl"
        self.flat_seconds = float(flat_seconds)
        self.flat_min_terminal_attempts = int(flat_min_terminal_attempts)
        self.minimum_run_age_seconds = float(minimum_run_age_seconds)
        self.planner = {
            "model": planner_model, "provider": planner_provider, "reasoning": planner_reasoning,
            "base_url": planner_base_url, "api_key_env": planner_api_key_env,
        }
        self.metric_subspace_seed = int(metric_subspace_seed)
        self.phase = "ordinary" if start_ordinary else "waiting_portfolio"
        self.cohort_index = -1 if start_ordinary else 0
        self._run_started_at = time.time()
        self.search: Any = None
        self._audit_id: str | None = None
        self._best_cycles: float | None = None
        self._progress_anchor_cycles: float | None = None
        self._flat_anchor_at: float | None = None
        self._scheduled_at: dict[str, float] = {}
        self._terminal_at: dict[str, float] = {}
        self._iteration_by_attempt: dict[str, int] = {}
        self._ordinary_attempts: set[str] = set()
        self._guided_iterations: set[int] = set()
        self._guided_terminal: set[int] = set()
        self._directions: list[dict[str, Any]] = []
        self._assigned_count = 0
        self._frozen_codes: dict[str, str] = {}
        self._planner_task: asyncio.Task | None = None
        self._planner_dirs: dict[int, Path] = {}
        self._failed: list[dict[str, Any]] = []

    # ---------- journal ----------

    def _journal(self, kind: str, **payload: Any) -> None:
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": time.time(), "kind": kind, "cohort_index": self.cohort_index,
                                     "audit_id": self._audit_id, **payload}, sort_keys=True) + "\n")

    # ---------- search hooks ----------

    def bind(self, search: Any) -> None:
        self.search = search
        self._best_cycles = search._cycles(search.best_id)
        self._progress_anchor_cycles = self._best_cycles
        self._flat_anchor_at = time.time()
        self._journal("controller_bound", phase=self.phase, native_concurrency=NATIVE_CONCURRENCY,
                      burst_capacity=32)
        if self.phase == "waiting_portfolio":
            # Fork: the first intervention plans from the exact branch cutoff.
            cutoff_seq = search.next_seq - 1
            if cutoff_seq < 0:
                raise RuntimeError("a branch start needs at least one completed attempt")
            self._request_portfolio(cutoff_seq)

    def on_scheduled(self, attempt_id: str, iteration: int, at: float,
                     guidance: dict[str, Any] | None) -> None:
        self._scheduled_at[attempt_id] = at
        self._iteration_by_attempt[attempt_id] = iteration
        if guidance is None:
            self._ordinary_attempts.add(attempt_id)
            return
        self._guided_iterations.add(iteration)
        self._journal("direction_assigned", generation=iteration, direction_index=guidance["direction_index"],
                      target_branch_id=guidance["target_branch_id"],
                      donor_branch_ids=guidance["donor_branch_ids"])

    def on_completion(self, *, seq: int, attempt_id: str, at: float, valid: bool, cycles: float | None,
                      child_id: str | None, guidance: dict[str, Any] | None) -> None:
        self._terminal_at[attempt_id] = at
        iteration = self._iteration_by_attempt.get(attempt_id, -1)
        guided = iteration in self._guided_iterations
        if guided:
            self._guided_terminal.add(iteration)
            self._journal("guided_attempt_terminal", generation=iteration, completion_seq=seq,
                          evaluation_valid=valid, child_id=child_id, cycles=cycles)
        if valid and cycles is not None:
            if self._best_cycles is None or cycles < self._best_cycles:
                self._best_cycles = cycles
            anchor = self._progress_anchor_cycles
            if anchor is None or (anchor - cycles) / abs(anchor) > MEANINGFUL_RELATIVE_GAIN:
                self._progress_anchor_cycles = cycles
                if self.phase == "ordinary":
                    self._flat_anchor_at = at
                self._journal("meaningful_frontier_improvement", completion_seq=seq, cycles=cycles,
                              source="guided" if guided else "ordinary")
        self._maybe_trigger(seq, at)
        if self._guided_iterations and self._guided_iterations <= self._guided_terminal:
            self._journal("guided_cohort_complete", guided_generations=sorted(self._guided_iterations))
            self._guided_iterations.clear()
            self._guided_terminal.clear()

    def _active_search_seconds(self, since: float, now: float) -> float:
        intervals = sorted(
            (max(start, since), min(self._terminal_at.get(attempt_id, now), now))
            for attempt_id, start in self._scheduled_at.items()
            if attempt_id in self._ordinary_attempts
            and min(self._terminal_at.get(attempt_id, now), now) > max(start, since)
        )
        total, cursor = 0.0, None
        for lo, hi in intervals:
            if cursor is None or lo > cursor:
                total += hi - lo
                cursor = hi
            elif hi > cursor:
                total += hi - cursor
                cursor = hi
        return total

    def _maybe_trigger(self, completion_seq: int, now: float) -> None:
        if (
            self.phase != "ordinary"
            or (self.minimum_run_age_seconds > 0 and now - self._run_started_at < self.minimum_run_age_seconds)
            or self._flat_anchor_at is None
            or self._guided_iterations
        ):
            return
        terminals = sum(at >= self._flat_anchor_at for attempt_id, at in self._terminal_at.items()
                        if attempt_id in self._ordinary_attempts)
        active_seconds = self._active_search_seconds(self._flat_anchor_at, now)
        if active_seconds < self.flat_seconds or terminals < self.flat_min_terminal_attempts:
            return
        self.cohort_index += 1
        self.phase = "waiting_portfolio"
        self._journal("repeat_flat_ready", completion_seq=completion_seq, active_search_seconds=active_seconds,
                      terminal_attempts_since_anchor=terminals,
                      progress_anchor_cycles=self._progress_anchor_cycles)
        self._request_portfolio(completion_seq)

    # ---------- planner ----------

    def _snapshot(self, checkpoint: Path) -> None:
        programs = self.search.programs
        checkpoint.mkdir(parents=True, exist_ok=False)
        for program in programs.values():
            _write_json(checkpoint / "programs" / f"{program['id']}.json", {
                "id": program["id"], "code": program["code"], "parent_id": program["parent_id"],
                "generation": program["generation"], "metrics": program["metrics"],
                "metadata": {"island": 0, "changes": program["changes"]},
            })
        _write_json(checkpoint / "metadata.json", {
            "best_program_id": self.search.best_id,
            "islands": [sorted(programs)],
            "island_best_programs": [self.search.best_id],
        })
        self._frozen_codes = {program_id: program["code"] for program_id, program in programs.items()}

    def _request_portfolio(self, cutoff_seq: int) -> None:
        if self.cohort_index > 0:
            previous = self.cohort_index - 1
            if previous in self._planner_dirs:
                self._failed = failed_directions(existing=self._failed, planner=self._planner_dirs[previous],
                                                 journal=self.journal_path, cohort=previous)
        checkpoint = self.control_dir / f"checkpoint_{self.cohort_index:03d}"
        self._snapshot(checkpoint)
        oe = load_oe_supervisor()
        events = [row for row in _rows(self.search.trace.path)]
        axis = json.loads(self.time_axis_path.read_text(encoding="utf-8"))
        normalized_events = oe._events_on_active_time_axis(events, axis)
        normalized = self.control_dir / f"attempts_{self.cohort_index:03d}.jsonl"
        normalized.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized_events),
                              encoding="utf-8")
        index = next(i for i, row in enumerate(events) if row.get("event") == "canonical_completion"
                     and row.get("completion_seq") == cutoff_seq)
        request = {
            "schema": "shinka_overlap_planner_request_v1",  # prepare_shinka_breadth_view.py input contract
            "framework": "metaharness",
            "cohort_index": self.cohort_index,
            "cutoff_seq": int(cutoff_seq),
            "cutoff_at": float(normalized_events[index]["timestamp"]),
            "source_cutoff_at": float(events[index]["timestamp"]),
            "checkpoint": str(checkpoint),
            "normalized_attempts": str(normalized),
        }
        request_path = self.control_dir / f"request_{self.cohort_index:03d}.json"
        _write_json(request_path, request)
        self._journal("planner_requested", **request)
        self._planner_task = asyncio.get_running_loop().create_task(self._plan(request_path, self.cohort_index))

    async def _run(self, command: list[str], *, log: Path, env: dict[str, str], timeout: float) -> None:
        with log.open("w", encoding="utf-8") as handle:
            process = await asyncio.create_subprocess_exec(
                *command, cwd=HERE, env=env, stdin=asyncio.subprocess.DEVNULL,
                stdout=handle, stderr=asyncio.subprocess.STDOUT,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"command failed ({process.returncode}); see {log}")

    async def _plan(self, request: Path, cohort: int) -> Path:
        epoch = self.epochs_dir / f"epoch_{cohort:03d}"
        epoch.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update({
            "AUDIT_PI_PROVIDER": self.planner["provider"], "AUDIT_PI_MODEL": self.planner["model"],
            "AUDIT_PI_REASONING": self.planner["reasoning"], "AUDIT_PI_API_KEY_ENV": self.planner["api_key_env"],
        })
        await self._run(
            [sys.executable, str(SH / "prepare_shinka_breadth_view.py"), "--request", str(request),
             "--output", str(epoch / "view_build"), "--prompt", str(PLANNER_PROMPT)],
            log=epoch / "prepare.log", env=env, timeout=3600.0,
        )
        failed_path = epoch / "prior_failed_interventions.json"
        _write_json(failed_path, self._failed)
        planner = epoch / "planner"
        await self._run(
            [sys.executable, str(OE / "run_branch_portfolio_breadth.py"),
             "--prepared-view", str(epoch / "view_build" / "prepared" / "bundle"),
             "--output", str(planner), "--prompt", str(PLANNER_PROMPT),
             "--candidate-pool-size", "32", "--directions-per-call", "8", "--planner-workers", "4",
             "--planner-random-metric-count", "15",
             "--metric-subspace-seed", str(self.metric_subspace_seed + cohort * 1000003),
             "--dedup-model", "deepseek-v4-flash", "--dedup-base-url", "https://api.deepseek.com",
             "--dedup-api-key-env", "DEEPSEEK_API_KEY", "--dedup-max-tokens", "128000",
             "--prior-failed-interventions", str(failed_path),
             "--model", self.planner["model"], "--provider", self.planner["provider"],
             "--base-url", self.planner["base_url"], "--api-key-env", self.planner["api_key_env"]],
            log=epoch / "planner.log", env=env, timeout=6 * 3600.0,
        )
        _write_json(self.control_dir / f"handoff_{cohort:03d}.json", {
            "schema": "shinka_overlap_portfolio_handoff_v1", "cohort_index": cohort, "planner_dir": str(planner),
        })
        return planner

    def poll(self) -> None:
        if self.phase != "waiting_portfolio" or self._planner_task is None or not self._planner_task.done():
            return
        task, self._planner_task = self._planner_task, None
        try:
            planner = task.result()
            self._load_portfolio(planner)
        except Exception as exc:  # a failed intervention returns control to native search
            self._journal("planner_failed", error=f"{type(exc).__name__}: {exc}")
            self._resume_ordinary()

    def _load_portfolio(self, planner: Path) -> None:
        portfolio = json.loads((planner / "portfolio.json").read_text(encoding="utf-8"))
        catalog = json.loads((planner / "bundle/workspace/meta/branch_catalog.json").read_text(encoding="utf-8"))
        catalog_by_id = {str(row["id"]): row for row in catalog["programs"]}
        directions = portfolio.get("directions")
        if not isinstance(directions, list) or not directions:
            raise RuntimeError("breadth planner returned no directions")
        for direction in directions:
            for branch_id in [str(direction["target_branch_id"]),
                              *[str(item) for item in direction.get("donor_branch_ids") or []]]:
                code = self._frozen_codes.get(branch_id)
                row = catalog_by_id.get(branch_id)
                if code is None or row is None or _sha256_text(code) != row.get("code_sha256"):
                    raise RuntimeError(f"selected frozen branch is invalid: {branch_id}")
        self._planner_dirs[self.cohort_index] = planner
        portfolio_sha = _canonical_sha(portfolio)
        self._audit_id = "mh-breadth-" + portfolio_sha[:16]
        diagnosis = portfolio.get("diagnosis") or "\n".join(str(item) for item in portfolio.get("diagnoses") or [])
        self._directions = []
        for index, direction in enumerate(directions, start=1):
            payload = {
                "plan_version": GUIDANCE_VERSION,
                "audit_id": self._audit_id,
                "diagnosis": diagnosis or None,
                "exhausted_patterns": list(portfolio.get("exhausted_patterns") or []),
                "lane_id": "intervention_direction",
                "lane_task": str(direction["task"]),
                "portfolio_sha256": portfolio_sha,
                "target_branch_id": str(direction["target_branch_id"]),
                "donor_branch_ids": [str(item) for item in direction.get("donor_branch_ids") or []],
                "diagnosed_bottleneck": str(direction["mechanism_hypothesis"]),
                "recipe": str(direction["task"]),
                "compatibility": [str(item) for item in direction["constraints"]],
                "direction_index": index,
            }
            payload["plan_hash"] = _canonical_sha({key: value for key, value in payload.items()
                                                   if key not in {"plan_hash", "direction_index"}})
            self._directions.append(payload)
        self._assigned_count = 0
        self.phase = "assigning_guided"
        self._journal("portfolio_ready", planner_dir=str(planner), direction_count=len(self._directions))
        self._journal("cohort_bound", direction_count=len(self._directions), frontier_cycles=self._best_cycles)

    # ---------- dispatch policy ----------

    def take_guidance(self) -> dict[str, Any] | None:
        if self.phase != "assigning_guided":
            return None
        payload = self._directions[self._assigned_count]
        self._assigned_count += 1
        if self._assigned_count == len(self._directions):
            self.phase = "burst_draining_to_native_width"
            self._journal("breadth_burst_fully_assigned", assigned_count=self._assigned_count)
        return payload

    def _resume_ordinary(self) -> None:
        self.phase = "ordinary"
        self._flat_anchor_at = time.time()
        self._ordinary_attempts = {attempt_id for attempt_id in self._ordinary_attempts
                                   if self._terminal_at.get(attempt_id, math.inf) >= self._flat_anchor_at}
        self._journal("ordinary_search_resumed")

    def ordinary_allowed(self, running: int) -> bool:
        if self.phase == "burst_draining_to_native_width" and running < NATIVE_CONCURRENCY:
            self._resume_ordinary()
        return self.phase == "ordinary"

    def busy(self) -> bool:
        return self.phase in {"waiting_portfolio", "assigning_guided"} or bool(self._guided_iterations)

    render_guidance = staticmethod(render_guidance)

    def shutdown(self) -> None:
        if self._planner_task is not None and not self._planner_task.done():
            self._planner_task.cancel()
        self._journal("controller_shutdown", phase=self.phase)
