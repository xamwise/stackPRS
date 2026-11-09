#!/usr/bin/env python3
"""Step 3 — compute the pretrained PRS with pgsc_calc, chunks in parallel.

Where the time goes (measured natively on an Apple Silicon Mac, 200 scores /
18.5 M scoring variants vs the 77k-variant UKB_CRC set; pgsc_calc's own tools):

  download scoring files   137 s   (sequential, ~2 MB/s)
  format scoring files     448 s   (2 threads; 12 threads: 315 s and +50% memory)
  match variants            40 s   (2 threads; peak 7.3 GB)
  plink2 genotype prep     0.1 s
  plink2 scoring           1.3 s   (12 threads: 0.7 s)

So the levers are parallel runs of 2 threads each, chunks balanced by variant
volume (step 02), and scoring files downloaded once — not more threads or a GPU.
This runner therefore

* sizes concurrency to the CPUs and memory Docker (or the host) actually has
  (``run.jobs: auto``, ``cpus_per_job``, ``memory_per_job``) and gives each run
  its slice via a Nextflow executor config + ``--max_cpus/--max_memory``;
* prefetches the harmonised scoring files in parallel into an md5-verified cache
  and hands each chunk its files via ``--scorefile`` (``run.prefetch_scorefiles``);
* runs each chunk in its OWN launch and work directory (concurrent Nextflow runs
  must not share ./work and ./.nextflow), after a one-off ``nextflow pull``;
* stamps results with their PGS ids and refuses stale ones; reports a chunk in
  which no score reaches --min_overlap as "no scores passed".

  python scripts/03_run_pgsc_calc.py                  # config defaults
  python scripts/03_run_pgsc_calc.py --jobs 3         # override concurrency
  python scripts/03_run_pgsc_calc.py --dry-run        # print the plan only
  python scripts/03_run_pgsc_calc.py --prefetch-only  # fill the scoring-file cache (e.g. before sbatch)

On an HPC cluster, use the Slurm array (scripts/03_run_pgsc_calc.slurm).
"""
import _bootstrap  # noqa: F401

import argparse
import concurrent.futures as cf
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

from stackprs import config, genotypes, scorefiles

PIPELINE = "pgscatalog/pgsc_calc"


def effective_profile(profile: str, machine: str | None = None) -> str:
    """Add pgsc_calc's ``arm`` profile for Docker on ARM hosts (Apple Silicon).

    Without it the amd64 containers run under emulation and variant matching
    segfaults (pgsc_calc docs, "How do I run the pipeline on an M1 Mac?").
    """
    parts = [p for p in profile.split(",") if p]
    machine = (machine or platform.machine()).lower()
    if "docker" in parts and "arm" not in parts and machine in ("arm64", "aarch64"):
        parts.append("arm")
    return ",".join(parts)


def machine_resources(profile: str) -> tuple[int, float, str]:
    """(CPUs, memory GB, where) available to the pipeline's processes.

    With Docker Desktop the containers only see its VM, whose CPU and memory
    limits are set in Docker's settings — often well below the host's.
    """
    if "docker" in profile.split(",") and shutil.which("docker"):
        res = subprocess.run(["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}}"],
                             capture_output=True, text=True)
        parts = res.stdout.split()
        # with the daemon down, `docker info --format` still exits 0 and prints "0 0"
        if res.returncode == 0 and len(parts) == 2 and all(p.isdigit() and int(p) > 0 for p in parts):
            return int(parts[0]), int(parts[1]) / 1e9, "Docker"
        print("WARNING: Docker is not reachable (is Docker Desktop running?); sizing from the host")
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    return os.cpu_count() or 1, mem, "host"


def plan_jobs(cpus: int, mem_gb: float, cpus_per_job: int, memory_per_job: float) -> int:
    """Concurrent runs that fit both the CPUs and (90% of) the memory."""
    return max(1, min(cpus // max(1, cpus_per_job), int(0.9 * mem_gb // memory_per_job)))


def resources_config(path: pathlib.Path, cpus: int, memory_gb: int) -> pathlib.Path:
    """Nextflow config giving one run its share of the machine (local executor)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("// written by stackPRS: this run's share of the machine (runs execute concurrently)\n"
                    f"executor {{\n    cpus   = {cpus}\n    memory = '{memory_gb} GB'\n}}\n")
    return path


def score_file(outdir: pathlib.Path, sampleset: str) -> pathlib.Path:
    """Where pgsc_calc v2 writes the scores for one run."""
    return outdir / sampleset / "score" / "aggregated_scores.txt.gz"


def build_command(cfg, chunk_file: pathlib.Path, *, profile: str, build: str, resume: bool,
                  cpus_per_job: int, memory_per_job: int, extra_config, min_overlap: float,
                  revision, scorefile_glob: str | None = None, pgs_ids: str | None = None) -> dict:
    """One pgsc_calc run. Scoring files come from `scorefile_glob` (prefetched) and/or
    `pgs_ids` (downloaded by pgsc_calc); without a glob all chunk ids go to --pgs_id."""
    chunk_id = chunk_file.stem
    outdir = cfg.results_dir / chunk_id
    launchdir = cfg.work_dir / "launch" / chunk_id
    ids = chunk_file.read_text().strip()
    if scorefile_glob is None:
        pgs_ids = ids
    cmd = [
        "nextflow", "run", PIPELINE,
        "-profile", profile,
        "-name", f"stackprs_{chunk_id}_{int(time.time())}",
        "-work-dir", str(cfg.work_dir / "work" / chunk_id),
        "-c", str(launchdir / "stackprs_resources.config"),
        "--input", str(cfg.work_dir / "samplesheet.csv"),
        "--target_build", build,
        "--min_overlap", str(min_overlap),
        "--max_cpus", str(cpus_per_job),
        "--max_memory", f"{memory_per_job}.GB",
        "--outdir", str(outdir),
    ]
    if scorefile_glob:
        cmd += ["--scorefile", scorefile_glob]
    if pgs_ids:
        cmd += ["--pgs_id", pgs_ids]
    if revision:
        cmd[3:3] = ["-r", revision]
    if extra_config:
        cmd += ["-c", str(extra_config)]
    if resume:
        cmd += ["-resume"]
    return {"chunk_id": chunk_id, "outdir": outdir, "ids": ids, "cmd": cmd, "launchdir": launchdir,
            "workdir": cfg.work_dir / "work" / chunk_id, "cpus": cpus_per_job, "memory": memory_per_job}


def stamp(outdir: pathlib.Path) -> pathlib.Path:
    return outdir / "pgs_ids.txt"


def is_done(plan: dict, sampleset: str) -> bool:
    return score_file(plan["outdir"], sampleset).exists()


def stale_chunks(plans: list[dict]) -> list[str]:
    """Chunks whose existing results were computed for a different set of PGS ids."""
    return [p["chunk_id"] for p in plans
            if stamp(p["outdir"]).exists() and stamp(p["outdir"]).read_text().strip() != p["ids"]]


def run_chunk(plan: dict, sampleset: str, logdir: pathlib.Path,
              genotype_fingerprint: str | None = None) -> tuple[str, str, float]:
    if is_done(plan, sampleset):
        return plan["chunk_id"], "skipped", 0.0
    if genotype_fingerprint is not None and genotypes.drop_stale_genotypes(plan["workdir"], genotype_fingerprint):
        print(f"  {plan['chunk_id']}: genotype files changed since the last run; dropped stale prepared genotypes")
    plan["launchdir"].mkdir(parents=True, exist_ok=True)
    resources_config(plan["launchdir"] / "stackprs_resources.config", plan["cpus"], plan["memory"])
    plan["outdir"].mkdir(parents=True, exist_ok=True)
    stamp(plan["outdir"]).write_text(plan["ids"])
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"{plan['chunk_id']}.log"
    t0 = time.time()
    with open(logfile, "w") as fh:
        # isolated launch dir -> each run gets its own .nextflow state
        rc = subprocess.run(plan["cmd"], cwd=plan["launchdir"], stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc == 0:
        status = "ok"
    elif "ZeroMatchesError" in logfile.read_text(errors="replace"):
        status = "no scores passed"
    else:
        status = f"FAILED(rc={rc})"
    return plan["chunk_id"], status, time.time() - t0


def prefetch(cfg, chunks: list[pathlib.Path], build: str, workers: int) -> dict[str, pathlib.Path]:
    """Download every chunk's scoring files once and stage one directory per chunk."""
    urls = scorefiles.load_harmonized_urls(cfg.cache_dir, build)
    ids = [i for c in chunks for i in c.read_text().strip().split(",") if i]
    t0 = time.time()
    files = scorefiles.fetch(ids, urls, cfg.work_dir / "scorefiles" / build, workers=workers)
    size = sum(p.stat().st_size for p in files.values()) / 1e9
    print(f"Scoring files: {len(files)}/{len(ids)} in the cache ({size:.2f} GB, {time.time() - t0:.0f}s)")
    stage_root = cfg.work_dir / "scorefiles" / "chunks"
    current = {c.stem for c in chunks}
    for old in stage_root.glob("chunk_*"):          # staging dirs of chunks that no longer exist
        if old.name not in current:
            shutil.rmtree(old)
    return {c.stem: scorefiles.stage_chunk(c.stem, c.read_text().strip().split(","), files, stage_root)
            for c in chunks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, help="concurrent chunk runs (default: run.jobs, 'auto' = fit the machine)")
    ap.add_argument("--profile", help="Nextflow profile (default: run.profile in config)")
    ap.add_argument("--target-build", help="override target genome build")
    ap.add_argument("--no-resume", action="store_true", help="do not pass -resume")
    ap.add_argument("--no-prefetch", action="store_true", help="let pgsc_calc download scoring files itself")
    ap.add_argument("--prefetch-only", action="store_true", help="fill the scoring-file cache and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and commands, run nothing")
    args = ap.parse_args()

    cfg = config.load()
    profile = effective_profile(args.profile or cfg.run_profile)
    build = args.target_build or cfg.build
    cpus_per_job = int(cfg.get("run.cpus_per_job", 2))
    memory_per_job = int(cfg.get("run.memory_per_job", 8))

    if not (cfg.work_dir / "samplesheet.csv").exists():
        raise SystemExit("Missing data/samplesheet.csv — run scripts/00_make_samplesheet.py first.")
    chunks = sorted(cfg.chunks_dir.glob("chunk_*.txt"))
    if not chunks:
        raise SystemExit("No chunks found — run scripts/02_filter_and_chunk.py first.")

    use_prefetch = args.prefetch_only or (cfg.get("run.prefetch_scorefiles", True) and not args.no_prefetch)
    stage_root = cfg.work_dir / "scorefiles" / "chunks"
    staged: dict[str, pathlib.Path] = {}
    if use_prefetch and not args.dry_run:
        staged = prefetch(cfg, chunks, build, int(cfg.get("run.download_workers", 8)))
        if args.prefetch_only:
            print(f"Staged {len(staged)} chunk(s) under {stage_root}")
            return

    cpus, mem_gb, where = machine_resources(profile)
    jobs_cfg = args.jobs or cfg.get("run.jobs", "auto")
    auto = plan_jobs(cpus, mem_gb, cpus_per_job, memory_per_job)
    jobs = auto if str(jobs_cfg).lower() == "auto" else int(jobs_cfg)
    if jobs > auto:
        print(f"WARNING: {jobs} concurrent runs x ({cpus_per_job} CPUs, {memory_per_job} GB) exceed what "
              f"{where} offers ({cpus} CPUs, {mem_gb:.0f} GB); expect swapping or killed containers")

    plans = []
    for c in chunks:
        if c.stem in staged:
            glob, missing = scorefiles.staged_inputs(staged[c.stem])
        elif use_prefetch and args.dry_run:               # what a real run would pass
            glob, missing = str(stage_root / c.stem / "*.txt.gz"), None
        else:
            glob, missing = None, None
        plans.append(build_command(cfg, c, profile=profile, build=build, resume=not args.no_resume,
                                   cpus_per_job=cpus_per_job, memory_per_job=memory_per_job,
                                   extra_config=cfg.extra_config, min_overlap=cfg.min_overlap,
                                   revision=cfg.pipeline_version, scorefile_glob=glob, pgs_ids=missing))

    stale = stale_chunks(plans)
    if stale:
        raise SystemExit(f"Results for {', '.join(stale)} were computed for a different set of scores "
                         f"(filters changed?). Delete {cfg.results_dir} to recompute.")

    print(f"{len(plans)} chunk(s) | {where}: {cpus} CPUs, {mem_gb:.0f} GB -> jobs={jobs} x "
          f"({cpus_per_job} CPUs, {memory_per_job} GB) | profile={profile} | build={build} | "
          f"min_overlap={cfg.min_overlap} | pgsc_calc={cfg.pipeline_version or 'latest pulled'} | "
          f"scoring files: {'prefetched' if use_prefetch else 'downloaded by pgsc_calc'}")
    host_mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    if where == "Docker" and auto < len(plans) and mem_gb < 0.75 * host_mem:
        print(f"  tip: Docker Desktop may use {mem_gb:.0f} of this Mac's {host_mem:.0f} GB; raising its memory "
              "limit (Settings > Resources) allows more concurrent runs")
    if args.dry_run:
        for p in plans:
            print(f"\n# {p['chunk_id']}  (cwd={p['launchdir']})\n  " + " ".join(p["cmd"]))
        return

    if not shutil.which("nextflow"):
        raise SystemExit("`nextflow` not found on PATH. Activate the conda env (environment.yml).")

    print(f"Warming pipeline cache: nextflow pull {PIPELINE} ...")
    pull = ["nextflow", "pull", PIPELINE] + (["-r", cfg.pipeline_version] if cfg.pipeline_version else [])
    subprocess.run(pull, check=False)

    logdir = cfg.work_dir / "logs"
    results: list[tuple[str, str, float]] = []
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:         # chunks are ordered heaviest-first
        fp = genotypes.fingerprint(cfg.work_dir / "samplesheet.csv")
        futures = [ex.submit(run_chunk, p, cfg.sampleset, logdir, fp) for p in plans]
        for fut in cf.as_completed(futures):
            cid, status, elapsed = fut.result()
            results.append((cid, status, elapsed))
            print(f"  [{len(results)}/{len(plans)}] {cid:<14} {status:<18} {elapsed:6.0f}s")

    failed = sorted(c for c, s, _ in results if s.startswith("FAILED"))
    empty = sorted(c for c, s, _ in results if s == "no scores passed")
    print(f"\nDone: {len(results) - len(failed) - len(empty)} ok/skipped, "
          f"{len(empty)} with no score passing min_overlap, {len(failed)} failed.")
    if empty:
        print(f"  no passing scores in: {', '.join(empty)} — variant overlap with your genotypes is "
              "too low (wrong target.build? array data? consider imputed genotypes)")
    if failed:
        print("  failed (see data/logs/<chunk>.log): " + ", ".join(failed))
        sys.exit(1)
    print(f"Results in {cfg.results_dir}/chunk_*/ — next: scripts/04_merge_scores.py")


if __name__ == "__main__":
    main()
