"""The parallel pgsc_calc runner, driven against a stub `nextflow`."""
import os
import stat

import pytest

from conftest import load_script

STUB = """#!/usr/bin/env bash
# Minimal stand-in for `nextflow run pgscatalog/pgsc_calc`: writes the v2 output
# layout, or fails like pgsc_calc does when no score reaches --min_overlap.
echo "$@" >> "$(dirname "$0")/calls.log"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir) out="$2"; shift ;;
    --pgs_id) ids="$2"; shift ;;
  esac
  shift
done
if [[ "$ids" == *PGS999999* ]]; then
  echo "pgscatalog.core.lib.pgsexceptions.ZeroMatchesError: All scores fail to meet match threshold 0.75"
  exit 1
fi
mkdir -p "$out/COH/score" && echo stub > "$out/COH/score/aggregated_scores.txt.gz"
"""


@pytest.fixture
def runner(make_cfg, tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "nextflow"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    cfg = make_cfg()
    (cfg.chunks_dir / "chunk_000.txt").write_text("PGS000001,PGS000002")
    (cfg.chunks_dir / "chunk_001.txt").write_text("PGS999999")
    return load_script("03_run_pgsc_calc"), cfg


def plans_for(mod, cfg, **kw):
    return [mod.build_command(cfg, c, profile="docker", build="GRCh38", resume=True, cpus_per_job=2,
                              memory_per_job=6, extra_config=None, min_overlap=0.75, revision="v2.3.0", **kw)
            for c in sorted(cfg.chunks_dir.glob("chunk_*.txt"))]


def arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_command_line(runner):
    mod, cfg = runner
    cmd = plans_for(mod, cfg)[0]["cmd"]
    assert cmd[:5] == ["nextflow", "run", "pgscatalog/pgsc_calc", "-r", "v2.3.0"]
    assert arg(cmd, "--min_overlap") == "0.75" and arg(cmd, "--target_build") == "GRCh38"
    assert arg(cmd, "--pgs_id") == "PGS000001,PGS000002" and "--scorefile" not in cmd
    assert arg(cmd, "--max_cpus") == "2" and arg(cmd, "--max_memory") == "6.GB"
    assert arg(cmd, "-c").endswith("launch/chunk_000/stackprs_resources.config")
    assert "-resume" in cmd


def test_prefetched_scorefiles_replace_pgs_id(runner):
    mod, cfg = runner
    cmd = plans_for(mod, cfg, scorefile_glob="/x/chunk_000/*.txt.gz", pgs_ids="")[0]["cmd"]
    assert arg(cmd, "--scorefile") == "/x/chunk_000/*.txt.gz" and "--pgs_id" not in cmd
    cmd = plans_for(mod, cfg, scorefile_glob="/x/*.txt.gz", pgs_ids="PGS000002")[0]["cmd"]
    assert arg(cmd, "--pgs_id") == "PGS000002"          # scores without a harmonised file


def test_arm_profile_added_only_for_docker_on_arm(runner):
    mod, _ = runner
    assert mod.effective_profile("docker", machine="arm64") == "docker,arm"
    assert mod.effective_profile("docker,arm", machine="arm64") == "docker,arm"
    assert mod.effective_profile("docker", machine="x86_64") == "docker"
    assert mod.effective_profile("singularity", machine="aarch64") == "singularity"


@pytest.mark.parametrize("cpus, mem, expected", [
    (12, 34.0, 5),    # memory-bound: 0.9 * 34 // 6 = 5 (CPU would allow 6)
    (12, 16.0, 2),    # a Docker VM with 16 GB
    (4, 64.0, 2),     # CPU-bound
    (1, 2.0, 1),      # never below one
])
def test_plan_jobs(runner, cpus, mem, expected):
    mod, _ = runner
    assert mod.plan_jobs(cpus, mem, cpus_per_job=2, memory_per_job=6) == expected


def test_statuses_skip_stale_and_resource_slice(runner):
    mod, cfg = runner
    plans = plans_for(mod, cfg)
    logs = cfg.work_dir / "logs"
    assert mod.run_chunk(plans[0], "COH", logs)[1] == "ok"
    assert mod.score_file(plans[0]["outdir"], "COH").exists()
    slice_cfg = (plans[0]["launchdir"] / "stackprs_resources.config").read_text()
    assert "cpus   = 2" in slice_cfg and "memory = '6 GB'" in slice_cfg
    assert mod.run_chunk(plans[1], "COH", logs)[1] == "no scores passed"
    assert mod.run_chunk(plans[0], "COH", logs)[1] == "skipped"
    assert mod.stale_chunks(plans) == []
    (cfg.chunks_dir / "chunk_000.txt").write_text("PGS000003")       # filters changed
    assert mod.stale_chunks(plans_for(mod, cfg)) == ["chunk_000"]


@pytest.mark.parametrize("docker_out, where", [("0 0", "host"), ("8 16000000000", "Docker")])
def test_machine_resources_falls_back_when_docker_daemon_is_down(runner, tmp_path, docker_out, where):
    """`docker info --format` exits 0 and prints zeros when the daemon is not running."""
    mod, _ = runner
    fake = tmp_path / "bin" / "docker"
    fake.write_text(f"#!/usr/bin/env bash\necho '{docker_out}'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    cpus, mem, got = mod.machine_resources("docker,arm")
    assert got == where and cpus > 0 and mem > 0
    if where == "Docker":
        assert (cpus, round(mem)) == (8, 16)
