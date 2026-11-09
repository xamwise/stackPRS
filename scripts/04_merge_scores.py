#!/usr/bin/env python3
"""Step 4 — merge per-chunk pgsc_calc output into one [N x scores] matrix.

Reads data/results/chunk_*/<sampleset>/score/aggregated_scores.txt.gz (long
format, PGS ids like PGS000018_hmPOS_GRCh37) and the per-score match summaries.
Writes:
  - data/pgs_feature_matrix.parquet (or .csv)   IID + one column per PGS accession
  - data/match_summary.csv                      per-score n_variants / match_rate / score_pass
"""
import _bootstrap  # noqa: F401

import pandas as pd

from stackprs import config, data


def main() -> None:
    cfg = config.load()
    ss = cfg.sampleset
    paths = sorted(cfg.results_dir.glob(f"chunk_*/{ss}/score/aggregated_scores.txt.gz"))
    if not paths:
        raise SystemExit(f"No {cfg.results_dir}/chunk_*/{ss}/score/aggregated_scores.txt.gz — run step 3 first.")

    print(f"Merging {len(paths)} chunk result(s) using the {cfg.score_value} value...")
    matrix = data.merge_score_chunks(paths, metric=cfg.score_value, sampleset=ss)

    summaries = sorted(cfg.results_dir.glob(f"chunk_*/{ss}/match/*_summary.csv"))
    if summaries:
        try:
            match = pd.concat([data.load_match_summary(p) for p in summaries], ignore_index=True)
        except (KeyError, ValueError) as e:     # optional QC input; format may differ between releases
            print(f"  WARNING: could not parse match summaries ({e}); skipping match-rate QC")
        else:
            match.to_csv(cfg.work_dir / "match_summary.csv", index=False)
            print(f"  match summary for {len(match)} scores (median match rate "
                  f"{match['match_rate'].median():.1%}; {int((~match['score_pass']).sum())} failed min_overlap)")

    # Only scores from the current filter set; report what pgsc_calc could not compute.
    clean_ids = pd.read_csv(cfg.work_dir / "pgs_metadata_clean.csv")["pgs_id"].tolist()
    computed = [c for c in matrix.columns if c != "IID"]
    extra = sorted(set(computed) - set(clean_ids))
    if extra:
        print(f"  dropping {len(extra)} score(s) not in pgs_metadata_clean.csv (stale?): {extra[:5]}")
    keep = [c for c in clean_ids if c in set(computed)]
    matrix = matrix[["IID"] + keep]
    not_scored = len(clean_ids) - len(keep)

    out = data.write_frame(matrix, cfg.work_dir / "pgs_feature_matrix")
    print(f"Wrote {out}")
    print(f"  individuals : {len(matrix)}")
    print(f"  scores      : {len(keep)} of {len(clean_ids)} selected "
          f"({not_scored} not scored — below min_overlap or failed chunks)")


if __name__ == "__main__":
    main()
