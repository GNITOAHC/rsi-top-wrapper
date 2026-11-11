#!/usr/bin/env python3
"""Run Meta-Harness with the repeated RSI-top breadth treatment.

Fresh start (no ``--source-trace-root``): ordinary search first, interventions
after the 1 h / 20 active flat min / 12 terminal attempts trigger.

OpenEvolve-style branch: copy a stopped baseline's causal trace prefix up to its
first flat cutoff (or ``--initial-completion-seq``), rebuild the archive from
it, and start with an intervention planned from that exact cutoff. Layout
matches the OpenEvolve treatment: supervisor files in ``--output``, the search
itself in ``--output/run``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mh_breadth  # noqa: E402
import mh_loop  # noqa: E402

ACTIVE_TIME_AXIS_SCHEMA = "oe_supervisor_active_time_axis_v1"
PREFIX_EVENTS = {"run_started", "attempt_scheduled", "canonical_completion", "program_committed", "db_state"}


def write_prefix(source_attempts: Path, cutoff_seq: int, target: Path) -> dict:
    """Copy the causal prefix byte for byte: every event of completions <= cutoff.

    ponytail: attempts still in flight at the cutoff are dropped, not replayed
    (OpenEvolve --preserve-inflight); none of them was committed, so the prefix
    stays causal. Upgrade path: re-dispatch them as ordinary attempts at start.
    """
    lines = source_attempts.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    completions = [row for row in rows if row.get("event") == "canonical_completion"
                   and row.get("completion_seq") == cutoff_seq]
    if len(completions) != 1:
        raise RuntimeError(f"cutoff {cutoff_seq} must match exactly one canonical completion")
    kept_attempts = {row["attempt_id"] for row in rows if row.get("event") == "canonical_completion"
                     and int(row["completion_seq"]) <= cutoff_seq}
    selected = []
    for line, row in zip(lines, rows):
        event = row.get("event")
        if event not in PREFIX_EVENTS:
            continue
        if event == "attempt_scheduled" and row.get("attempt_id") not in kept_attempts:
            continue
        if event not in {"run_started", "attempt_scheduled"} and int(row["completion_seq"]) > cutoff_seq:
            continue
        selected.append(line)
    data = b"".join(selected)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {
        "schema": "mh_branch_manifest_v1",
        "source_attempts": str(source_attempts),
        "source_attempts_sha256": hashlib.sha256(source_attempts.read_bytes()).hexdigest(),
        "completion_seq": cutoff_seq,
        "cutoff_at": float(completions[0]["timestamp"]),
        "historical_prefix_sha256": hashlib.sha256(data).hexdigest(),
        "historical_prefix_events": len(selected),
        "dropped_inflight_attempts": sorted(
            {row["attempt_id"] for row in rows if row.get("event") == "attempt_scheduled"}
            - kept_attempts
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-trace-root", type=Path, help="stopped Meta-Harness baseline search directory")
    parser.add_argument("--initial-completion-seq", type=int)
    parser.add_argument("--budget", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-provider", choices=["deepseek"], default="deepseek")
    parser.add_argument("--worker-model", default="deepseek-v4-flash")
    parser.add_argument("--worker-reasoning", default="high")
    parser.add_argument("--wall-time-seconds", type=float, default=12 * 3600.0)
    parser.add_argument("--minimum-run-age-seconds", type=float, default=3600.0)
    parser.add_argument("--flat-seconds", type=float, default=1200.0)
    parser.add_argument("--flat-min-terminal-attempts", type=int, default=12)
    parser.add_argument("--planner-model", default="deepseek-v4-flash")
    parser.add_argument("--planner-provider", default="deepseek")
    parser.add_argument("--planner-reasoning", default="high")
    parser.add_argument("--planner-base-url", default="https://api.deepseek.com")
    parser.add_argument("--planner-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--metric-subspace-seed", type=int, default=20260830)
    parser.add_argument("--pi", default=mh_loop.os.environ.get("AUDIT_PI_CLI", "pi"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started_at = time.time()
    args = parse_args(argv)
    pi = mh_loop.check_environment(args.pi)
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    if args.initial_completion_seq is not None and args.source_trace_root is None:
        raise SystemExit("--initial-completion-seq requires --source-trace-root")
    output.mkdir(parents=True)
    oe = mh_breadth.load_oe_supervisor()
    manifest = {
        "schema": "mh_repeat_flat_breadth_manifest_v1",
        "argv": sys.argv,
        "created_at": started_at,
        "dispatch_mode": "overlap",
        "minimum_run_age_seconds": args.minimum_run_age_seconds,
        "trigger": {"flat_seconds": args.flat_seconds, "terminal_attempts": args.flat_min_terminal_attempts,
                    "meaningful_relative_gain": mh_breadth.MEANINGFUL_RELATIVE_GAIN},
        "planner": {"model": args.planner_model, "provider": args.planner_provider,
                    "reasoning": args.planner_reasoning, "calls": 4, "directions_per_call": 8,
                    "metrics_per_call": 15},
        "credential_values_recorded": False,
    }
    prefix = None
    if args.source_trace_root is None:
        manifest.update(source_trace_root=None, fresh_start=True)
        mh_loop.write_json(output / "supervisor_manifest.json", manifest)
        mh_loop.write_json(output / "active_time_axis.json", {
            "schema": ACTIVE_TIME_AXIS_SCHEMA, "source_ended_at": started_at,
            "segments": [{"raw_started_at": started_at, "active_started_seconds": 0.0}],
        })
    else:
        source = args.source_trace_root.resolve()
        attempts = source / "attempts.jsonl"
        if not attempts.is_file():
            raise SystemExit(f"source trace not found: {attempts}")
        if args.initial_completion_seq is None:
            cutoff = oe.first_flat_cutoff(
                attempts, minimum_elapsed_seconds=args.minimum_run_age_seconds,
                flat_seconds=args.flat_seconds, flat_min_terminal_attempts=args.flat_min_terminal_attempts,
            )
        else:
            cutoff = {"completion_seq": args.initial_completion_seq, "selection": "explicit"}
        mh_loop.write_json(output / "initial_cutoff.json", cutoff)
        manifest.update(source_trace_root=str(source), fresh_start=False,
                        source_attempts_sha256=hashlib.sha256(attempts.read_bytes()).hexdigest())
        mh_loop.write_json(output / "supervisor_manifest.json", manifest)
        prefix = output / "initial_branch" / "historical_prefix.jsonl"
        mh_loop.write_json(output / "initial_branch" / "branch_manifest.json",
                           write_prefix(attempts, int(cutoff["completion_seq"]), prefix))
        oe._event_time_axis(output=output, historical_prefix=prefix, supervisor_started_at=started_at,
                            active_started_seconds=0.0, resume_existing=False)
    run = output / "run"
    breadth = mh_breadth.BreadthManager(
        output=run, time_axis_path=output / "active_time_axis.json", epochs_dir=output / "breadth_epochs",
        flat_seconds=args.flat_seconds, flat_min_terminal_attempts=args.flat_min_terminal_attempts,
        minimum_run_age_seconds=args.minimum_run_age_seconds, start_ordinary=prefix is None,
        planner_model=args.planner_model, planner_provider=args.planner_provider,
        planner_reasoning=args.planner_reasoning, planner_base_url=args.planner_base_url,
        planner_api_key_env=args.planner_api_key_env, metric_subspace_seed=args.metric_subspace_seed,
    )
    search = mh_loop.Search(
        output=run, budget=args.budget, worker_model=args.worker_model, worker_reasoning=args.worker_reasoning,
        pi=pi, wall_time_seconds=args.wall_time_seconds, seed=args.seed, breadth=breadth,
    )
    search.setup(prefix)
    asyncio.run(search.run())
    mh_loop.write_json(output / "run_summary.json", {
        "schema": "mh_repeat_flat_breadth_summary_v1",
        "finished_at": time.time(),
        "wall_time_seconds": args.wall_time_seconds,
        "best_program_id": search.best_id,
        "best_cycles": search._cycles(search.best_id),
        "completed_attempts": len(search.summary),
        "interventions": breadth.cohort_index + 1,
        "finish_guided_cohort_after_wall_time": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
