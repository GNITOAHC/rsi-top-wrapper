#!/usr/bin/env python3
"""Export one "Anthropic kernel optimization" results panel from finished runs.

Prints one ``FRONTIER`` entry and one ``COST`` entry in the project site's
format: best score so far (``1363 / cycles``) against hours, for a baseline run
and the RSI-top run forked from it, plus intervention dots (planner delivered),
baseline stall intervals, and optional dashed-line reference runs. Works on any
release run whose trace uses the shared attempt schema.

    uv run python export_frontier.py --name "Meta-Harness · DeepSeek v4 Flash" \\
        --baseline runs/mh-baseline --rsi runs/mh-rsi [--gpt RUN] [--opus RUN]
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OE = ROOT / "integrations" / "openevolve-takehome"
NORMALIZER = 1363.0
MEANINGFUL_RELATIVE_GAIN = 0.002
FLASH_PRICE = (0.14, 0.28)  # deepseek-v4-flash USD per 1M input/output tokens (dedup calls)
DELIVERED = {"portfolio_ready", "overlap_portfolio_ready"}


@functools.lru_cache(maxsize=None)
def _axis_normalizer():
    if str(OE) not in sys.path:
        sys.path.insert(0, str(OE))
    spec = importlib.util.spec_from_file_location("oe_repeat_flat_breadth", OE / "run_repeat_flat_breadth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._events_on_active_time_axis


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _search(run: Path) -> Path:
    for candidate in (run / "search" / "run", run / "search", run):
        if (candidate / "attempts.jsonl").is_file():
            return candidate
    raise FileNotFoundError(f"no attempts.jsonl under {run}")


def _axis(run: Path) -> dict | None:
    for path in (run / "search" / "active_time_axis.json", run / "active_time_axis.json"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _normalize(rows: list[dict[str, Any]], axis: dict | None, key: str = "timestamp") -> list[dict[str, Any]]:
    if axis is None:
        return rows
    shifted = _axis_normalizer()([{**row, "timestamp": row[key]} for row in rows], axis)
    return [{**row, key: moved["timestamp"]} for row, moved in zip(rows, shifted)]


def _cycles(row: dict[str, Any]) -> float | None:
    """Valid cycles of one completion (OpenEvolve, Shinka, or Meta-Harness trace)."""
    child = row.get("child") or {}
    valid = row.get("evaluation_valid") is True or (row.get("evaluation_valid") is None and child.get("correct"))
    metrics = child.get("metrics") or child.get("public_metrics") or {}
    value = metrics.get("cycles")
    if valid and isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


class Trace:
    def __init__(self, run: Path) -> None:
        rows = _normalize(_rows(_search(run) / "attempts.jsonl"), _axis(run))
        self.started = min(row["timestamp"] for row in rows if row.get("event") == "run_started")
        first = next(row for row in rows if row.get("event") == "run_started")
        root = (first.get("root_metrics") or {}).get("cycles") or next(
            (row["parent_metrics"]["cycles"] for row in rows if row.get("event") == "attempt_scheduled"
             and isinstance((row.get("parent_metrics") or {}).get("cycles"), (int, float))), 147734.0)
        self.root_cycles = float(root)
        self.completions = sorted(
            (row for row in rows if row.get("event") == "canonical_completion"),
            key=lambda row: row["completion_seq"],
        )

    def hours(self, timestamp: float) -> float:
        return (timestamp - self.started) / 3600.0

    def best_points(self) -> list[tuple[float, float]]:
        """(hour, best cycles) at every frontier improvement, starting at the root."""
        points, best = [(0.0, self.root_cycles)], self.root_cycles
        for row in self.completions:
            cycles = _cycles(row)
            if cycles is not None and cycles < best:
                best = cycles
                points.append((self.hours(row["timestamp"]), best))
        return points


def best_at(points: list[tuple[float, float]], hour: float) -> float:
    return min(cycles for at, cycles in points if at <= hour)


def step_path(points: list[tuple[float, float]], start: float, end: float) -> list[list[float]]:
    """Site step format: a horizontal then vertical segment for every improvement."""
    current = best_at(points, start)
    path = [[round(start, 3), round(NORMALIZER / current, 4)]]
    for at, cycles in points:
        if start < at <= end and cycles < current:
            path.append([round(at, 3), round(NORMALIZER / current, 4)])
            current = cycles
            path.append([round(at, 3), round(NORMALIZER / current, 4)])
    path.append([round(end, 3), round(NORMALIZER / current, 4)])
    return path


def stalls(trace: Trace, horizon: float, after: float = 1.0) -> list[list[float]]:
    """Flat stretches between meaningful (>0.2%) improvements, after the first hour."""
    anchor, moves = trace.root_cycles, []
    for row in trace.completions:
        cycles = _cycles(row)
        if cycles is not None and (anchor - cycles) / abs(anchor) > MEANINGFUL_RELATIVE_GAIN:
            anchor = cycles
            moves.append(trace.hours(row["timestamp"]))
    edges = [after, *[at for at in moves if after < at < horizon], horizon]
    return [[round(lo, 3), round(hi, 3)] for lo, hi in zip(edges, edges[1:]) if hi > lo]


def label(cycles: float) -> str:
    return f"{cycles:g}"


def cost_until(baseline: Trace, rsi: Trace, rsi_run: Path, fork_hour: float, horizon: float) -> float | None:
    def attempts(trace: Trace, lo: float, hi: float) -> float | None:
        costs = [row.get("api_cost_usd") for row in trace.completions if lo <= trace.hours(row["timestamp"]) <= hi]
        if any(not isinstance(cost, (int, float)) for cost in costs):
            return None
        return float(sum(costs))

    worker = [attempts(baseline, 0.0, fork_hour), attempts(rsi, fork_hour + 1e-9, horizon)]
    if None in worker:
        return None
    planner = 0.0
    search = rsi_run / "search"
    journal = next((p for p in (_search(rsi_run) / "breadth_interventions.jsonl",
                                _search(rsi_run) / "portfolio_cohort.jsonl") if p.is_file()), None)
    delivered = {row["cohort_index"]: row for row in _normalize(_rows(journal), _axis(rsi_run), "at")
                 if row.get("kind") in DELIVERED} if journal else {}
    for run_json in sorted([*search.glob("breadth_epochs/epoch_*/planner/run.json"),
                            *search.glob("epochs/epoch_*/planner/run.json")]):
        cohort = int(run_json.parents[1].name.removeprefix("epoch_"))
        row = delivered.get(cohort)
        if row is None or rsi.hours(row["at"]) > horizon:
            continue
        run = json.loads(run_json.read_text(encoding="utf-8"))
        dedup = run.get("dedup_usage") or {}
        planner += float((run.get("usage") or {}).get("catalog_cost_estimate_usd") or 0.0)
        planner += (int(dedup.get("input_tokens") or 0) * FLASH_PRICE[0]
                    + int(dedup.get("output_tokens") or 0) * FLASH_PRICE[1]) / 1e6
    return sum(worker) + planner


def export(*, name: str, baseline_run: Path, rsi_run: Path, horizon: float,
           gpt_run: Path | None = None, opus_run: Path | None = None, reference_budget: float = 300.0) -> dict:
    baseline, rsi = Trace(baseline_run), Trace(rsi_run)
    cutoff = json.loads((rsi_run / "search" / "initial_cutoff.json").read_text(encoding="utf-8"))
    fork_row = next(row for row in baseline.completions if row["completion_seq"] == int(cutoff["completion_seq"]))
    fork = baseline.hours(fork_row["timestamp"])
    van_points, ours_points = baseline.best_points(), rsi.best_points()
    journal = next(p for p in (_search(rsi_run) / "breadth_interventions.jsonl",
                               _search(rsi_run) / "portfolio_cohort.jsonl") if p.is_file())
    dots = []
    for row in _normalize(_rows(journal), _axis(rsi_run), "at"):
        hour = rsi.hours(row["at"])
        if row.get("kind") in DELIVERED and hour <= horizon:
            dots.append([round(hour, 3), round(NORMALIZER / best_at(ours_points, hour), 4)])
    entry: dict[str, Any] = {
        "name": name,
        "H": horizon,
        "fork": round(fork, 3),
        "van": step_path(van_points, 0.0, horizon),
        "ours": step_path(ours_points, fork, horizon),
        "dots": dots,
        "labels": {"van": label(best_at(van_points, horizon)), "ours": label(best_at(ours_points, horizon))},
        "stalls": stalls(baseline, horizon),
    }
    for key, run in (("gpt", gpt_run), ("opus", opus_run)):
        if run is not None:
            final = min(cycles for _, cycles in Trace(run).best_points())
            entry[key] = round(NORMALIZER / final, 4)
            entry["labels"][key] = label(final)
    spent = cost_until(baseline, rsi, rsi_run, fork, horizon)
    cost = None if spent is None else {
        "r": f"{reference_budget / spent:.1f}×", "d": f"${spent:.0f} vs. ${reference_budget:.0f}",
        "spent_usd": round(spent, 2),
    }
    return {"frontier": entry, "cost": cost}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rsi", type=Path, required=True)
    parser.add_argument("--gpt", type=Path, help="reference run with GPT-5.6 Sol")
    parser.add_argument("--opus", type=Path, help="reference run with Claude Opus 5")
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--reference-budget", type=float, default=300.0)
    args = parser.parse_args(argv)
    result = export(name=args.name, baseline_run=args.baseline, rsi_run=args.rsi, horizon=args.hours,
                    gpt_run=args.gpt, opus_run=args.opus, reference_budget=args.reference_budget)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
