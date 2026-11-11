"""Offline Meta-Harness integration checks: trace contract, fork, breadth, export."""
import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.helpers import ROOT, load_integration

mh_loop = load_integration("release_mh_loop", "integrations/metaharness-takehome/mh_loop.py")
mh_breadth = load_integration("release_mh_breadth", "integrations/metaharness-takehome/mh_breadth.py")
mh_entry = load_integration("release_mh_entry", "integrations/metaharness-takehome/run_repeat_flat_breadth.py")
prepare_view = load_integration("release_shinka_prepare", "integrations/shinka-takehome/prepare_shinka_breadth_view.py")
export_frontier = load_integration("release_export_frontier", "export_frontier.py")
ROOT_CODE = (ROOT / "integrations/openevolve-takehome/initial_program.py").read_text()


def fake_cycles(iteration: int) -> int:
    return 1500 if iteration < 3 else 1400  # early gains, then a flat frontier


def fake_evaluate(path):
    match = re.search(r"# cycles=(\d+)", Path(path).read_text())
    cycles = float(match.group(1)) if match else 147734.0
    return True, {"combined_score": 0.5, "cycles": cycles, "speedup": 147734 / cycles}, {}


def install_fakes(monkeypatch, *, invalid_every=5):
    async def fake_pi(self, workspace, message):
        await asyncio.sleep(0.01)
        iteration = int(workspace.name)
        out = workspace / "view" / "out"
        if iteration % invalid_every == invalid_every - 1:
            return {"input_tokens": 10, "output_tokens": 5, "successful_read_paths": [], "returncode": 1}
        base = "root" if iteration < 4 else json.loads(
            (workspace / "view" / "frontier.json").read_text())["best_program_id"]
        (out / "candidate.py").write_text(ROOT_CODE + f"\n# cycles={fake_cycles(iteration)} v{iteration}\n")
        (out / "pending.json").write_text(json.dumps({"base_id": base, "changes": f"variant {iteration}"}))
        return {"input_tokens": 1000, "cache_read_tokens": 500, "output_tokens": 200,
                "successful_read_paths": [f"candidates/{base}.py", "frontier.json"], "returncode": 0,
                "tool_calls": 3}

    monkeypatch.setattr(mh_loop.Search, "_run_pi", fake_pi)
    monkeypatch.setattr(mh_loop, "evaluate_program", fake_evaluate)


def run_search(output: Path, *, seconds: float, breadth=None, prefix=None):
    search = mh_loop.Search(output=output, budget=100.0, worker_model="deepseek-v4-flash",
                            worker_reasoning="high", pi="pi", wall_time_seconds=seconds, seed=1,
                            breadth=breadth)
    search.setup(prefix)
    asyncio.run(search.run())
    return search


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    install_fakes(monkeypatch)
    output = tmp_path / "mh-baseline" / "search"
    search = run_search(output, seconds=0.6)
    return search, output


def test_trace_feeds_shared_history_dashboard_and_cutoff(baseline):
    search, output = baseline
    events = rows(output / "attempts.jsonl")
    completions = [row for row in events if row["event"] == "canonical_completion"]
    assert len(completions) >= 12
    assert [row["completion_seq"] for row in completions] == list(range(len(completions)))
    assert any(row["terminal"] == "worker_error" for row in completions)
    commits = [row for row in events if row["event"] == "program_committed"]
    assert {row["program_id"] for row in commits} | {"root"} == set(search.programs)
    assert all(row["api_cost_usd"] > 0 for row in completions if row["terminal"] != "abandoned_at_shutdown")
    assert search.best_id != "root" and search._cycles(search.best_id) == 1400.0
    assert (output / "best/best_program.py").read_text() == search.programs[search.best_id]["code"]

    from planner_context import build_stage_context
    last = completions[-1]
    context = build_stage_context(
        attempts_path=output / "attempts.jsonl", cutoff_seq=last["completion_seq"],
        cutoff_at=last["timestamp"], root_code=ROOT_CODE, live_program_ids=set(search.programs),
    )
    assert "search_history/dashboard_llm.jsonl" in context["files"]

    oe = mh_breadth.load_oe_supervisor()
    cutoff = oe.first_flat_cutoff(output / "attempts.jsonl", minimum_elapsed_seconds=0.0,
                                  flat_seconds=0.05, flat_min_terminal_attempts=4)
    assert 0 < cutoff["completion_seq"] <= last["completion_seq"]


def test_fork_prefix_is_causal_and_rebuilds_the_archive(baseline, tmp_path, monkeypatch):
    search, output = baseline
    cutoff = 6
    prefix = tmp_path / "branch" / "historical_prefix.jsonl"
    manifest = mh_entry.write_prefix(output / "attempts.jsonl", cutoff, prefix)
    source_lines = set((output / "attempts.jsonl").read_bytes().splitlines(keepends=True))
    prefix_lines = prefix.read_bytes().splitlines(keepends=True)
    assert set(prefix_lines) <= source_lines  # byte-for-byte copies
    prefix_rows = [json.loads(line) for line in prefix_lines]
    assert max(row.get("completion_seq", -1) for row in prefix_rows) == cutoff
    kept = {row["attempt_id"] for row in prefix_rows if row["event"] == "canonical_completion"}
    assert {row["attempt_id"] for row in prefix_rows if row["event"] == "attempt_scheduled"} == kept
    assert manifest["completion_seq"] == cutoff

    forked = mh_loop.Search(output=tmp_path / "fork", budget=1.0, worker_model="deepseek-v4-flash",
                            worker_reasoning="high", pi="pi", wall_time_seconds=1.0, seed=1)
    forked.setup(prefix)
    committed = {row["program_id"] for row in prefix_rows if row["event"] == "program_committed"}
    assert set(forked.programs) == committed | {"root"}
    assert forked.next_seq == cutoff + 1
    assert (tmp_path / "fork/attempts.jsonl").read_bytes().startswith(prefix.read_bytes())


def fake_planner_factory(directions=32):
    async def fake_plan(self, request, cohort):
        planner = self.epochs_dir / f"epoch_{cohort:03d}" / "planner"
        catalog = planner / "bundle/workspace/meta/branch_catalog.json"
        catalog.parent.mkdir(parents=True)
        catalog.write_text(json.dumps({"programs": [
            {"id": key, "code_sha256": mh_breadth._sha256_text(code)} for key, code in self._frozen_codes.items()
        ]}))
        target = json.loads(Path(json.loads(request.read_text())["checkpoint"]).joinpath(
            "metadata.json").read_text())["best_program_id"]
        (planner / "portfolio.json").write_text(json.dumps({
            "diagnosis": "flat", "exhausted_patterns": [],
            "directions": [{"target_branch_id": target, "mechanism_hypothesis": f"h{i}", "task": f"t{i}",
                            "constraints": ["stay correct"], "donor_branch_ids": []} for i in range(directions)],
        }))
        (planner / "run.json").write_text(json.dumps({
            "usage": {"catalog_cost_estimate_usd": 0.5}, "dedup_usage": {"input_tokens": 1000, "output_tokens": 10},
        }))
        return planner
    return fake_plan


def test_breadth_trigger_dispatches_whole_cohort_and_resumes(tmp_path, monkeypatch):
    install_fakes(monkeypatch, invalid_every=1000)
    monkeypatch.setattr(mh_breadth.BreadthManager, "_plan", fake_planner_factory())
    search_dir = tmp_path / "mh-rsi" / "search"
    search_dir.mkdir(parents=True)
    now = mh_loop.time.time()
    (search_dir / "active_time_axis.json").write_text(json.dumps({
        "schema": "oe_supervisor_active_time_axis_v1", "source_ended_at": now,
        "segments": [{"raw_started_at": now, "active_started_seconds": 0.0}]}))
    breadth = mh_breadth.BreadthManager(
        output=search_dir / "run", time_axis_path=search_dir / "active_time_axis.json",
        epochs_dir=search_dir / "breadth_epochs", flat_seconds=0.05, flat_min_terminal_attempts=4,
        start_ordinary=True,
    )
    search = run_search(search_dir / "run", seconds=2.0, breadth=breadth)
    journal = rows(search_dir / "run/breadth_interventions.jsonl")
    kinds = [row["kind"] for row in journal]
    for kind in ("repeat_flat_ready", "planner_requested", "portfolio_ready", "cohort_bound",
                 "breadth_burst_fully_assigned", "guided_cohort_complete", "ordinary_search_resumed"):
        assert kind in kinds
    first = [row for row in journal if row["cohort_index"] == 0]
    assert sum(row["kind"] == "direction_assigned" for row in first) == 32
    assert sum(row["kind"] == "guided_attempt_terminal" for row in first) == 32
    request = json.loads((search_dir / "run/breadth_control/request_000.json").read_text())
    assert Path(request["normalized_attempts"]).is_file()
    guided = [row for row in rows(search_dir / "run/attempts.jsonl")
              if row["event"] == "canonical_completion" and row["meta_intervention"]]
    assert len(guided) >= 32 and all(row["parent_id"] == guided[0]["parent_id"] for row in guided[:32])

    # The frozen request is a valid input to the shared planner view (unchanged Shinka path).
    view = prepare_view.prepare(request=search_dir / "run/breadth_control/request_000.json",
                                output=tmp_path / "view", prompt=ROOT / "prompts/planner.md")
    catalog = json.loads((view / "workspace/meta/branch_catalog.json").read_text())
    assert {row["id"] for row in catalog["programs"]} == set(
        json.loads(Path(request["checkpoint"]).joinpath("metadata.json").read_text())["islands"][0])
    assert (view / "search_history/dashboard_llm.jsonl").is_file()


def test_failed_directions_memory_matches_shinka(tmp_path):
    planner = tmp_path / "planner"
    planner.mkdir()
    (planner / "portfolio.json").write_text(json.dumps({"directions": [
        {"task": "a", "mechanism_hypothesis": "x", "target_branch_id": "p1"}]}))
    journal = tmp_path / "journal.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in [
        {"kind": "cohort_bound", "cohort_index": 0, "frontier_cycles": 1400.0},
        {"kind": "direction_assigned", "cohort_index": 0, "generation": 7, "direction_index": 1},
        {"kind": "guided_attempt_terminal", "cohort_index": 0, "generation": 7, "evaluation_valid": True,
         "cycles": 1450.0, "completion_seq": 9},
        {"kind": "guided_cohort_complete", "cohort_index": 0},
    ]))
    ours = mh_breadth.failed_directions(existing=[], planner=planner, journal=journal, cohort=0)
    from tests.helpers import shinka_supervisor
    theirs = shinka_supervisor._failed_directions(existing=[], planner=planner, journal=journal, cohort=0)
    assert ours == theirs and ours[0]["outcome"] == "regress"


def test_export_frontier_produces_site_entry(baseline, tmp_path, monkeypatch):
    search, output = baseline
    rsi = tmp_path / "mh-rsi"
    (rsi / "search").mkdir(parents=True)
    cutoff = 8
    prefix = rsi / "search/initial_branch/historical_prefix.jsonl"
    mh_entry.write_prefix(output / "attempts.jsonl", cutoff, prefix)
    (rsi / "search/initial_cutoff.json").write_text(json.dumps({"completion_seq": cutoff}))
    (rsi / "search/supervisor_manifest.json").write_text(json.dumps({
        "created_at": mh_loop.time.time(), "source_trace_root": str(output)}))
    oe = mh_breadth.load_oe_supervisor()
    oe._event_time_axis(output=rsi / "search", historical_prefix=prefix,
                        supervisor_started_at=mh_loop.time.time(), active_started_seconds=0.0,
                        resume_existing=False)
    install_fakes(monkeypatch, invalid_every=1000)
    monkeypatch.setattr(mh_breadth.BreadthManager, "_plan", fake_planner_factory(directions=4))
    breadth = mh_breadth.BreadthManager(
        output=rsi / "search/run", time_axis_path=rsi / "search/active_time_axis.json",
        epochs_dir=rsi / "search/breadth_epochs", flat_seconds=10.0, start_ordinary=False,
    )
    run_search(rsi / "search/run", seconds=0.4, breadth=breadth, prefix=prefix)

    result = export_frontier.export(name="Meta-Harness · DeepSeek v4 Flash", baseline_run=output.parent,
                                    rsi_run=rsi, horizon=12.0, gpt_run=output.parent)
    entry = result["frontier"]
    assert set(entry) == {"name", "H", "fork", "van", "ours", "dots", "labels", "stalls", "gpt"}
    assert entry["van"][0][0] == 0.0 and entry["van"][-1][0] == 12.0
    assert entry["ours"][0][0] == entry["fork"] and entry["ours"][-1][0] == 12.0
    assert len(entry["dots"]) == 1 and entry["dots"][0][0] >= entry["fork"]
    assert entry["labels"]["van"] == "1400" and entry["gpt"] == round(1363 / 1400, 4)
    assert all(hi > lo for lo, hi in entry["stalls"])
    assert result["cost"]["spent_usd"] > 0 and result["cost"]["d"].endswith("vs. $300")


@pytest.mark.skipif(shutil.which("node") is None, reason="node unavailable")
def test_candidate_write_tool_only_writes_fixed_outputs(tmp_path):
    script = tmp_path / "harness.mjs"
    extension = ROOT / "integrations/metaharness-takehome/pi_candidate_write_extension.ts"
    script.write_text(
        f"import register from {json.dumps(str(extension))};\n"
        "let tool, hook; register({registerTool(t) { tool = t; }, on(name, h) { hook = h; }});\n"
        "console.log('max_tokens', hook({payload: {model: 'm'}}).max_tokens);\n"
        "await tool.execute('1', {code: 'x = 1\\n', base_id: '../../etc', changes: 'c'});\n"
        "try { await tool.execute('2', {code: 'y'.repeat(60001), base_id: 'root', changes: 'c'}); "
        "console.log('accepted-long'); } catch { console.log('rejected-long'); }\n"
    )
    result = subprocess.run(["node", "--experimental-strip-types", str(script)], cwd=tmp_path,
                            capture_output=True, text=True)
    if "ERR_UNKNOWN_FILE_EXTENSION" in result.stderr:
        pytest.skip("node cannot load TypeScript")
    assert result.returncode == 0, result.stderr
    assert "rejected-long" in result.stdout and "max_tokens 128000" in result.stdout
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["candidate.py", "pending.json"]
    assert (tmp_path / "out/candidate.py").read_text() == "x = 1\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["harness.mjs", "out"]
