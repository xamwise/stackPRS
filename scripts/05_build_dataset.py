#!/usr/bin/env python3
"""Step 5 — assemble the modelling dataset.

Joins the score matrix to phenotype + covariates, applies score QC, then
z-scores each PGS within the cohort (label-free). Writes:
  - data/model_dataset.parquet   IID, phenotype (0/1), covariates, standardised PGS
  - data/dataset_columns.json    which columns are PGS features / covariates
  - data/feature_qc.csv          per-score completeness / variance / match rate / kept
  - data/feature_scaling.csv     mean + sd used to standardise each PGS (to score new people)

Score QC: completeness across individuals (--min-complete), non-zero variance,
and — if data/match_summary.csv exists — the pgsc_calc variant match rate
(dataset.min_match_rate). Genotyping-array data (e.g. the bundled ~77k-variant
UKB_CRC set) match few variants of genome-wide scores, so inspect feature_qc.csv.
"""
import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd

from stackprs import config, data
from stackprs.learners import column_aucs


def main() -> None:
    cfg = config.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-complete", type=float, default=float(cfg.get("dataset.min_complete", 0.90)),
                    help="keep a score only if this fraction of individuals have a value")
    ap.add_argument("--min-match-rate", type=float, default=float(cfg.get("dataset.min_match_rate", 0.0)),
                    help="keep a score only if at least this fraction of its variants matched")
    args = ap.parse_args()

    matrix = data.read_frame(cfg.work_dir / "pgs_feature_matrix")
    df, pgs_cols, cov_cols = data.build_dataset(matrix, cfg.pheno, cfg.covariate, pheno_col=cfg.pheno_col)
    y = df[cfg.pheno_col]
    if len(df) == 0 or y.nunique() < 2:
        raise SystemExit(f"Need cases and controls after joining; got N={len(df)}, "
                         f"levels={sorted(y.unique())}. Check IIDs and target.pheno_col.")
    print(f"Joined cohort: {len(df)} individuals ({int(y.sum())} cases), {len(pgs_cols)} scores, "
          f"covariates={cov_cols}")

    # --- per-score QC ---
    qc = pd.DataFrame({"completeness": df[pgs_cols].notna().mean(),
                       "variance": df[pgs_cols].var()})
    qc.index.name = "pgs_id"
    ok = (qc["completeness"] >= args.min_complete) & (qc["variance"] > 0)
    match_path = cfg.work_dir / "match_summary.csv"
    if match_path.exists():
        match = pd.read_csv(match_path).drop_duplicates("pgs_id").set_index("pgs_id")
        qc = qc.join(match[["n_variants", "match_rate"]], how="left")
        ok &= qc["match_rate"].fillna(0) >= args.min_match_rate
    qc["kept"] = ok
    qc.to_csv(cfg.work_dir / "feature_qc.csv")
    keep = qc.index[ok].tolist()
    print(f"Score QC: kept {len(keep)} / {len(pgs_cols)} (completeness >= {args.min_complete:.0%}, "
          f"variance > 0, match rate >= {args.min_match_rate:.0%})")
    if not keep:
        raise SystemExit("No scores left after QC — see data/feature_qc.csv")

    # --- confounding check: covariates such as genetic PCs should not separate cases from controls ---
    cov_auc = column_aucs(df[cov_cols], y) if cov_cols else pd.Series(dtype=float)
    flagged = cov_auc[(cov_auc - 0.5).abs() > 0.15]
    for c, a in flagged.items():
        print(f"WARNING: covariate {c} alone separates cases from controls (AUC {max(a, 1 - a):.3f}). "
              "For genetic PCs this points to case/control batch or ancestry differences: PRS that "
              "correlate with it look predictive for the wrong reason, while residualising on it "
              "(dataset.residualize_on) also removes genuine signal. Treat raw AUCs as upper and "
              "residualised AUCs as lower bounds, and fix sampling/QC upstream.")

    # --- mean-impute residual missingness, z-score within the cohort (label-free) ---
    raw = df[keep]
    mean = raw.mean()
    imputed = raw.fillna(mean)
    sd = imputed.std(ddof=0)
    feats = (imputed - mean) / sd
    scaling = pd.DataFrame({"mean": mean, "sd": sd})

    # --- optional: remove the part of each PRS explained by covariates (e.g. PCs), label-free ---
    adjust = list(cfg.get("dataset.residualize_on", []) or [])
    unknown = [c for c in adjust if c not in cov_cols]
    if unknown:
        raise SystemExit(f"dataset.residualize_on: {unknown} not among covariates {cov_cols}")
    if adjust:
        A = np.column_stack([np.ones(len(df)), df[adjust].to_numpy(dtype=float)])
        B, *_ = np.linalg.lstsq(A, feats.to_numpy(), rcond=None)
        resid = feats.to_numpy() - A @ B
        r2 = 1 - resid.var(axis=0) / feats.to_numpy().var(axis=0)
        resid_sd = resid.std(axis=0)
        feats = pd.DataFrame(resid / resid_sd, columns=keep, index=df.index)
        scaling = scaling.join(pd.DataFrame(B.T, index=keep,
                                            columns=["resid_intercept"] + [f"resid_{c}" for c in adjust]))
        scaling["resid_sd"] = resid_sd
        print(f"Residualised PGS on {adjust}: variance explained median {np.median(r2):.1%}, max {r2.max():.1%}")
    scaling.rename_axis("pgs_id").to_csv(cfg.work_dir / "feature_scaling.csv")

    out = pd.concat([df[["IID", cfg.pheno_col] + cov_cols], feats], axis=1)
    path = data.write_frame(out, cfg.work_dir / "model_dataset")
    (cfg.work_dir / "dataset_columns.json").write_text(json.dumps(
        {"id": "IID", "phenotype": cfg.pheno_col, "pgs": keep, "covariates": cov_cols,
         "residualized_on": adjust}, indent=1))

    print(f"Wrote {path}")
    print(f"  N={len(out)}  cases={int(y.sum())}  controls={int((y == 0).sum())}  PGS features={len(keep)}")
    if not np.isfinite(feats.to_numpy()).all():
        raise SystemExit("Non-finite values after standardisation — inspect data/feature_qc.csv")


if __name__ == "__main__":
    main()
