"""Checks on the target genotypes: which genome build are they in?

pgsc_calc matches scoring-file variants to the target by chromosome:position,
so ``target.build`` must be right. Scoring GRCh38 genotypes with GRCh37
coordinates matches almost nothing: in UKB_CRC, 2 of the 144 variants shared
with PGS000193 sit at the GRCh37 position versus 144/144 at GRCh38 — and every
score then fails ``--min_overlap``.

The build is detected by comparing the target's chr:pos set with the harmonised
positions of a few small reference scores (variants spread genome-wide) in both
builds. This needs no rsIDs in the target and only a few hundred kB of downloads,
which are cached.
"""
from __future__ import annotations

import gzip
import pathlib

import requests

from . import catalog

BUILDS = ("GRCh37", "GRCh38")
REFERENCE_SCORES = ("PGS000193", "PGS000137", "PGS000509")   # 1-3k variants each, genome-wide
HARMONIZED_URL = ("https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{pgs}/ScoringFiles/"
                  "Harmonized/{pgs}_hmPOS_{build}.txt.gz")


def _norm_chr(c: str) -> str:
    c = str(c).strip()
    return c[3:] if c.lower().startswith("chr") else c


def target_positions(prefix: pathlib.Path, fmt: str = "bfile") -> set[tuple[str, int]]:
    """(chromosome, position) of every target variant (.bim or uncompressed .pvar)."""
    if fmt == "bfile":
        path, chr_col, pos_col, sep_header = prefix.parent / f"{prefix.name}.bim", 0, 3, False
    elif fmt == "pfile":
        path, chr_col, pos_col, sep_header = prefix.parent / f"{prefix.name}.pvar", 0, 1, True
    else:
        raise ValueError(f"Build detection supports bfile/pfile, not {fmt!r}")
    out: set[tuple[str, int]] = set()
    with open(path) as fh:
        for line in fh:
            if sep_header and line.startswith("#"):
                continue
            f = line.split()
            out.add((_norm_chr(f[chr_col]), int(f[pos_col])))
    return out


def reference_positions(pgs_id: str, build: str, cache_dir: pathlib.Path,
                        session: requests.Session | None = None) -> set[tuple[str, int]]:
    """Harmonised (hm_chr, hm_pos) of one PGS Catalog score in `build` (cached)."""
    cache = cache_dir / f"{pgs_id}_hmPOS_{build}.txt.gz"
    if not cache.exists():
        r = (session or catalog.make_session()).get(HARMONIZED_URL.format(pgs=pgs_id, build=build), timeout=120)
        r.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)
    out: set[tuple[str, int]] = set()
    with gzip.open(cache, "rt") as fh:
        header = None
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if header is None:
                header = {name: i for i, name in enumerate(f)}
                continue
            c, p = f[header["hm_chr"]], f[header["hm_pos"]]
            if c and p and p != "NA":
                out.add((_norm_chr(c), int(float(p))))
    return out


def detect_build(prefix: pathlib.Path, fmt: str, cache_dir: pathlib.Path,
                 session: requests.Session | None = None,
                 min_hits: int = 20, min_ratio: float = 5.0) -> tuple[str | None, dict[str, int]]:
    """Return (build or None if undecided, {build: matching positions})."""
    target = target_positions(prefix, fmt)
    hits = {}
    for build in BUILDS:
        ref: set[tuple[str, int]] = set()
        for pgs in REFERENCE_SCORES:
            ref |= reference_positions(pgs, build, cache_dir, session)
        hits[build] = len(target & ref)
    best, other = sorted(BUILDS, key=hits.get, reverse=True)
    if hits[best] >= min_hits and hits[best] >= min_ratio * max(1, hits[other]):
        return best, hits
    return None, hits


# --------------------------------------------------------------------------- #
# .bim compatibility with pgsc_calc
# --------------------------------------------------------------------------- #
def bim_has_genetic_distances(bim: pathlib.Path) -> bool:
    """True if any variant has a non-zero centimorgan value (column 3).

    pgsc_calc v2.3.0's matcher (pgscatalog.match 0.4.0) parses that column as
    an unsigned integer and aborts on values like 0.490722 ("could not parse
    ... as dtype u8"), although the column plays no role in scoring.
    """
    with open(bim) as fh:
        for line in fh:
            f = line.split(None, 3)
            if len(f) > 2 and f[2] not in ("0", "0.0"):
                return True
    return False


def zero_genetic_distances(prefix: pathlib.Path, out_prefix: pathlib.Path) -> pathlib.Path:
    """Write `<out_prefix>.bim` with the cM column set to 0; hard-link .bed/.fam.

    The input files are never modified. Hard links avoid copying the genotypes;
    across file systems the .bed/.fam are copied instead. Work is only redone
    when a source file changed, so the copy's fingerprint stays stable and
    pgsc_calc's prepared genotypes remain reusable between runs.
    """
    import os
    import shutil
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    src = prefix.parent / prefix.name
    src_bim, dst_bim = pathlib.Path(f"{src}.bim"), pathlib.Path(f"{out_prefix}.bim")
    if not (dst_bim.exists() and dst_bim.stat().st_mtime_ns >= src_bim.stat().st_mtime_ns):
        tmp = dst_bim.with_name(dst_bim.name + ".part")
        with open(src_bim) as fin, open(tmp, "w") as fout:
            for line in fin:
                f = line.rstrip("\n").split()
                f[2] = "0"
                fout.write("\t".join(f) + "\n")
        tmp.replace(dst_bim)
    for ext in (".bed", ".fam"):
        s, dst = pathlib.Path(f"{src}{ext}"), pathlib.Path(f"{out_prefix}{ext}")
        if dst.exists() and (os.path.samefile(s, dst) or dst.stat().st_mtime_ns >= s.stat().st_mtime_ns):
            continue
        dst.unlink(missing_ok=True)
        try:
            os.link(s, dst)
        except OSError:
            shutil.copy2(s, dst)
    return out_prefix


# --------------------------------------------------------------------------- #
# Stale prepared genotypes in pgsc_calc work directories
# --------------------------------------------------------------------------- #
_GENO_EXTS = {"bfile": (".bed", ".bim", ".fam"), "pfile": (".pgen", ".pvar", ".psam"), "vcf": (".vcf.gz",)}


def fingerprint(samplesheet: pathlib.Path) -> str:
    """Identity of the genotype files a samplesheet points to (path, size, mtime).

    pgsc_calc keeps its prepared ("relabelled") genotypes in
    ``<work-dir>/genomes/`` via Nextflow ``storeDir``, which reuses outputs by
    file *name* alone: after the genotype files change, a run in an existing
    work directory silently scores against the old variant list. Comparing this
    fingerprint lets the runner drop those stale files first.
    """
    import csv
    import os
    parts = []
    with open(samplesheet, newline="") as fh:
        for row in csv.DictReader(fh):
            for ext in _GENO_EXTS.get(row["format"], ("",)):
                f = os.path.realpath(f"{row['path_prefix']}{ext}")
                st = os.stat(f) if os.path.exists(f) else None
                parts.append(f"{f}:{st.st_size}:{st.st_mtime_ns}" if st else f"{f}:missing")
    return "\n".join(parts) + "\n"


def drop_stale_genotypes(workdir: pathlib.Path, current: str) -> bool:
    """Delete ``<workdir>/genomes`` unless it was prepared from `current`; record it."""
    import shutil
    stamp = workdir / "genotype_fingerprint.txt"
    stale = (workdir / "genomes").exists() and (not stamp.exists() or stamp.read_text() != current)
    if stale:
        shutil.rmtree(workdir / "genomes")
    workdir.mkdir(parents=True, exist_ok=True)
    stamp.write_text(current)
    return stale
