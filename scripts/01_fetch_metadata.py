#!/usr/bin/env python3
"""Step 1 — enumerate every score in the PGS Catalog with its metadata.

Writes:
  - data/all_pgs_metadata.csv          one row per score (the "score dictionary")
  - data/cache/pgs_scores.json.gz      raw API records, for audit / re-derivation

The metadata carries the cohort evidence used by step 02 (structured cohort and
free-text mentions for each cohort in catalog.KNOWN_COHORTS, e.g. UK Biobank
and All of Us, plus all contributing cohort names) and ancestry
distributions. It is also joined back onto model coefficients in step 07.

  python scripts/01_fetch_metadata.py             # download (~1 min)
  python scripts/01_fetch_metadata.py --offline   # rebuild from the cached records
"""
import _bootstrap  # noqa: F401

import argparse
import gzip
import json

import pandas as pd

from stackprs import catalog, config

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_):  # type: ignore
        return it


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="re-derive the metadata from data/cache/pgs_scores.json.gz (no download)")
    args = ap.parse_args()

    cfg = config.load()
    cache = cfg.cache_dir / "pgs_scores.json.gz"
    if args.offline:
        if not cache.exists():
            raise SystemExit(f"No cached records at {cache} — run without --offline once")
        with gzip.open(cache, "rt") as fh:
            raw = json.load(fh)
    else:
        raw = list(tqdm(catalog.iter_scores(cfg.api_base, cfg.page_size),
                        desc="fetching PGS metadata", unit="score"))
        with gzip.open(cache, "wt") as fh:
            json.dump(raw, fh)

    df = pd.DataFrame([catalog.flatten_score(s) for s in raw])
    out = cfg.work_dir / "all_pgs_metadata.csv"
    df.to_csv(out, index=False)

    print(f"\nWrote {out}")
    print(f"  total scores                  : {len(df)}")
    for spec in catalog.KNOWN_COHORTS.values():
        p = spec.key.lower()
        text_only = (~df[f"{p}_cohort"]) & df[f"{p}_text_evidence"].str.len().gt(0)
        print(f"  {spec.key + ' listed as cohort':<30}: {int(df[f'{p}_cohort'].sum())}")
        print(f"  {spec.key + ' only in free text':<30}: {int(text_only.sum())}")
    print(f"  no cohort information         : {int(df['cohorts'].fillna('').eq('').sum())}")
    print(f"  median n_variants             : {df['n_variants'].median():.0f}")


if __name__ == "__main__":
    main()
