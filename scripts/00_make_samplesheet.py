#!/usr/bin/env python3
"""Write the pgsc_calc samplesheet (data/samplesheet.csv) from pipeline.yaml.

Also writes data/pgsc_params.env, which the Slurm script sources, so shell code
never has to parse YAML. pgsc_calc resolves paths relative to the Nextflow
launch directory, so the PLINK prefix is written as an absolute path.

Genotype checks (see stackprs/genotypes.py):
  * `target.build` must match the genotypes — a wrong build makes pgsc_calc match
    almost no variants (skip with --no-build-check, e.g. offline);
  * a .bim with centimorgan values crashes pgsc_calc v2.3.0's matcher, so pgsc_calc
    gets a copy with cM = 0 in data/genotypes/ (hard links; inputs untouched);
  * data/genotype_fingerprint.txt identifies the genotype files, so the runners can
    drop pgsc_calc's stale prepared genotypes when they change.
"""
import _bootstrap  # noqa: F401

import argparse
import csv
import shlex

from stackprs import config, genotypes


def check_build(cfg) -> None:
    try:
        build, hits = genotypes.detect_build(cfg.plink_prefix, cfg.plink_format, cfg.cache_dir / "build_check")
    except Exception as e:     # offline, unsupported format, ...
        print(f"WARNING: could not check the genome build ({type(e).__name__}: {e}); "
              f"make sure target.build ({cfg.build}) is right")
        return
    counts = ", ".join(f"{b}: {n}" for b, n in hits.items())
    if build is None:
        print(f"WARNING: genome build undetermined from reference positions ({counts}); "
              f"make sure target.build ({cfg.build}) is right")
    elif build != cfg.build:
        raise SystemExit(f"Your genotypes are in {build}, but target.build is {cfg.build} "
                         f"(reference positions found — {counts}). Set target.build: {build} in "
                         "config/pipeline.yaml; otherwise pgsc_calc matches almost no variants.")
    else:
        print(f"Genome build check: genotypes are {build} (reference positions found — {counts})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-build-check", action="store_true", help="skip the genome-build check")
    args = ap.parse_args()

    cfg = config.load()
    prefix = cfg.plink_prefix
    if cfg.plink_format == "bfile":
        # NB: append the extension as a string — Path.with_suffix would replace
        # the trailing ".QC" in a prefix like "UKB_CRC.QC".
        for ext in (".bed", ".bim", ".fam"):
            sib = prefix.parent / f"{prefix.name}{ext}"
            if not sib.exists():
                raise FileNotFoundError(f"Missing {sib} — check paths.plink_prefix in pipeline.yaml")

    if not args.no_build_check:     # before writing anything: a wrong build must not reach pgsc_params.env
        check_build(cfg)

    geno = prefix                   # what pgsc_calc reads
    if cfg.plink_format == "bfile" and genotypes.bim_has_genetic_distances(prefix.parent / f"{prefix.name}.bim"):
        geno = genotypes.zero_genetic_distances(prefix, cfg.work_dir / "genotypes" / prefix.name)
        print(f"Note: {prefix.name}.bim has centimorgan positions, which pgsc_calc v2.3.0's variant matcher "
              f"cannot parse; pgsc_calc gets a copy with cM = 0 (scores are unaffected):\n  {geno}")

    out = cfg.work_dir / "samplesheet.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sampleset", "path_prefix", "chrom", "format"])
        # blank `chrom` = all chromosomes are in a single fileset
        w.writerow([cfg.sampleset, str(geno), "", cfg.plink_format])

    # identity of the genotype files: lets the runners drop stale prepared genotypes
    (cfg.work_dir / "genotype_fingerprint.txt").write_text(genotypes.fingerprint(out))

    params = {
        "SAMPLESHEET": out,
        "SAMPLESET": cfg.sampleset,
        "TARGET_BUILD": cfg.build,
        "MIN_OVERLAP": cfg.min_overlap,
        "PIPELINE_VERSION": cfg.pipeline_version or "",
    }
    env = cfg.work_dir / "pgsc_params.env"
    env.write_text("".join(f"{k}={shlex.quote(str(v))}\n" for k, v in params.items()))

    print(f"Wrote {out}")
    print(f"  sampleset = {cfg.sampleset}")
    print(f"  prefix    = {geno}  ({cfg.plink_format})")
    print(f"Wrote {env} (sourced by scripts/03_run_pgsc_calc.slurm)")
    spaced = [str(p) for p in (geno, cfg.work_dir) if " " in str(p)]
    if spaced:
        print("WARNING: path(s) contain spaces, which Nextflow/container mounts may not handle:\n  "
              + "\n  ".join(spaced) + "\n  If pgsc_calc fails, use a space-free path (e.g. a symlink).")


if __name__ == "__main__":
    main()
