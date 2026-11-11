import itertools
import json
import subprocess
import sys

import pytest

from tests.helpers import ROOT


@pytest.mark.parametrize("method,framework,provider", [
    *itertools.product(["baseline", "rsi-top"], ["openevolve", "shinka"], ["deepseek", "gemini"]),
    ("baseline", "metaharness", "deepseek"), ("rsi-top", "metaharness", "deepseek")])
def test_public_dry_run_is_credential_free_and_does_not_write(tmp_path, method, framework, provider):
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), method, "--framework", framework,
         "--provider", provider, "--output", str(output), "--dry-run"],
        capture_output=True, text=True, check=True, cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
    )
    spec, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert spec["framework"] == framework and spec["provider"] == provider
    if method == "rsi-top":
        assert "--minimum-run-age-seconds" in spec["command"]
    assert not output.exists()


def dry_run(tmp_path, *args):
    return subprocess.run([sys.executable, str(ROOT / "run.py"), *args, "--dry-run"],
                          capture_output=True, text=True, cwd=tmp_path, env={"PATH": "/usr/bin:/bin"})


def test_metaharness_fork_dry_run_targets_the_baseline_search(tmp_path):
    source = tmp_path / "mh-baseline"
    (source / "search").mkdir(parents=True)
    (source / "launch.json").write_text(json.dumps({
        "method": "baseline", "framework": "metaharness", "provider": "deepseek", "model": "deepseek-v4-flash"}))
    result = dry_run(tmp_path, "rsi-top", "--framework", "metaharness", "--output", str(tmp_path / "rsi"),
                     "--source", str(source), "--cutoff", "40")
    spec, _ = json.JSONDecoder().raw_decode(result.stdout)
    command = spec["command"]
    assert command[1].endswith("metaharness-takehome/run_repeat_flat_breadth.py")
    assert command[command.index("--source-trace-root") + 1] == str((source / "search").resolve())
    assert command[command.index("--initial-completion-seq") + 1] == "40"
    assert "--initial-ordinary" not in command and "--config" not in command


def test_metaharness_rejects_other_providers(tmp_path):
    result = dry_run(tmp_path, "baseline", "--framework", "metaharness", "--provider", "gemini",
                     "--output", str(tmp_path / "run"))
    assert result.returncode == 2 and "deepseek only" in result.stderr
