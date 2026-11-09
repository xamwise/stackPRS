#!/usr/bin/env python3
"""Step 7 — which pretrained PRS drive the disease-specific ensemble?

Reads the final model (step 06) and reports:
  - data/ensemble/top_scores.csv        per PGS: uniLasso (Python/R) + lasso coefficients,
                                        univariate AUC, AutoGluon permutation importance
                                        (held-out folds), trait metadata, target/related flags
  - data/ensemble/unilasso_agreement.json   do the R and Python uniLasso agree?
  - data/ensemble/efo_enrichment.csv    trait (EFO) enrichment of each learner's selected
                                        PGS vs. all PGS in the model — the pleiotropy readout
  - data/figures/top_scores.png

uniLasso keeps each selected PGS's coefficient sign equal to its univariate
association, so a selected score of another trait is a directly interpretable
pleiotropy signal.
"""
import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

from stackprs import config, data
from stackprs.learners import column_aucs


def efo_enrichment(selected: set[str], background: list[str], meta: pd.DataFrame,
                   min_size: int = 2) -> pd.DataFrame:
    """Hypergeometric enrichment of EFO trait labels among selected PGS.

    Background = PGS actually used as model features. All labels with at least
    ``min_size`` background scores are tested (also those with 0 selected), and
    Benjamini-Hochberg FDR is applied across them.
    """
    m = meta[meta["pgs_id"].isin(background)].copy()
    m["efo"] = m["trait_efo_labels"].fillna("").str.split("|")
    ex = m.explode("efo")
    ex = ex[ex["efo"].str.len() > 0]
    M, N = len(set(background)), len(selected)
    rows = []
    for efo, grp in ex.groupby("efo"):
        ids = set(grp["pgs_id"])
        if len(ids) < min_size:
            continue
        k = len(ids & selected)
        rows.append({"efo_label": efo, "n_background": len(ids), "n_selected": k,
                     "expected": N * len(ids) / M,
                     "p_value": float(hypergeom.sf(k - 1, M, len(ids), N)) if k else 1.0})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("p_value").reset_index(drop=True)
    p = out["p_value"].to_numpy()
    ranked = p * len(p) / np.arange(1, len(p) + 1)
    out["fdr"] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])   # Benjamini-Hochberg
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30, help="rows to show/plot")
    args = ap.parse_args()

    cfg = config.load()
    final = cfg.ensemble_dir / "final"
    coef = pd.read_csv(final / "coefficients.csv", index_col="feature").drop(index="(intercept)")
    cols = json.loads((cfg.work_dir / "dataset_columns.json").read_text())
    meta = pd.read_csv(cfg.work_dir / "pgs_metadata_clean.csv")
    ds = data.read_frame(cfg.work_dir / "model_dataset")
    pgs = cols["pgs"]

    table = pd.DataFrame(index=pd.Index(pgs, name="pgs_id"))
    table["univariate_auc"] = column_aucs(ds[pgs], ds[cols["phenotype"]])
    table = table.join(coef.add_prefix("coef_"), how="left")
    coef_cols = [c for c in table.columns if c.startswith("coef_")]
    table[coef_cols] = table[coef_cols].fillna(0.0) + 0.0      # + 0.0 turns -0.0 (theta * beta) into 0.0
    table["n_learners_selected"] = (table[coef_cols] != 0).sum(axis=1)

    fi_path = cfg.ensemble_dir / "cv_autogluon_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path).groupby("feature")["importance"].mean().rename("autogluon_importance")
        table = table.join(fi, how="left")

    keep_meta = ["pgs_id", "trait_reported", "trait_efo_labels", "first_author", "n_variants",
                 "gwas_ancestry", "is_target_trait", "is_related_trait"]
    table = table.reset_index().merge(meta[[c for c in keep_meta if c in meta]], on="pgs_id", how="left")
    # consensus first: selected by more learners, then larger mean |coef|, then univariate strength
    table = (table.assign(_mean_abs=table[coef_cols].abs().mean(axis=1) if coef_cols else 0.0,
                          _uni=(table["univariate_auc"] - 0.5).abs())
                  .sort_values(["n_learners_selected", "_mean_abs", "_uni"], ascending=False)
                  .drop(columns=["_mean_abs", "_uni"]))
    table.to_csv(cfg.ensemble_dir / "top_scores.csv", index=False)

    stacker = json.loads((final / "stacker.json").read_text())
    # plot the linear learner the ensemble relies on most
    lead = max((c for c in coef_cols), key=lambda c: stacker["shares"].get(c.removeprefix("coef_"), 0), default=None)

    # --- R vs Python uniLasso agreement ---
    if {"coef_unilasso_py", "coef_unilasso_r"} <= set(table.columns):
        a, b = table["coef_unilasso_py"].fillna(0), table["coef_unilasso_r"].fillna(0)
        sa, sb = set(table.pgs_id[a != 0]), set(table.pgs_id[b != 0])
        union = (a != 0) | (b != 0)
        agree = {"selected_python": len(sa), "selected_r": len(sb), "selected_both": len(sa & sb),
                 "jaccard": len(sa & sb) / len(sa | sb) if sa | sb else 1.0,
                 "spearman_coef_on_union": float(spearmanr(a[union], b[union])[0]) if union.sum() > 2 else None,
                 "sign_agreement_on_both": float(np.mean(np.sign(a[(a != 0) & (b != 0)]) ==
                                                         np.sign(b[(a != 0) & (b != 0)]))) if sa & sb else None}
        (cfg.ensemble_dir / "unilasso_agreement.json").write_text(json.dumps(agree, indent=1))
        print(f"uniLasso Python vs R: {agree['selected_python']} vs {agree['selected_r']} PGS selected, "
              f"{agree['selected_both']} shared (Jaccard {agree['jaccard']:.2f})")

    # --- EFO enrichment per selector ---
    enr = []
    for c in coef_cols:
        sel = set(table.loc[table[c].fillna(0) != 0, "pgs_id"])
        if sel:
            e = efo_enrichment(sel, pgs, meta)
            if not e.empty:
                enr.append(e.assign(selector=c.removeprefix("coef_")))
    if enr:
        enr = pd.concat(enr, ignore_index=True)
        enr.to_csv(cfg.ensemble_dir / "efo_enrichment.csv", index=False)

    print("Ensemble learner shares: " + ", ".join(f"{k}={v:.0%}" for k, v in stacker["shares"].items()))
    show = ["pgs_id", "trait_reported", "univariate_auc", "n_learners_selected"] + coef_cols + \
           [c for c in ("autogluon_importance", "is_target_trait") if c in table]
    print(f"\nTop {args.top} PGS:")
    print(table.head(args.top)[show].to_string(index=False, float_format="%.3f"))
    if isinstance(enr, pd.DataFrame) and not enr.empty:
        print("\nMost enriched trait labels among selected PGS (FDR < 0.1):")
        sig = enr[enr["fdr"] < 0.1]
        print(sig.head(20).to_string(index=False, float_format="%.3g") if not sig.empty else "  none")

    if lead:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        sel = table[table[lead] != 0]
        plot = sel.loc[sel[lead].abs().sort_values(ascending=False).index].head(args.top).iloc[::-1]
        if plot.empty:
            return
        labels = plot["trait_reported"].fillna("").astype(str).str.slice(0, 45) + "  (" + plot["pgs_id"] + ")"
        fig, ax = plt.subplots(figsize=(9, 0.34 * len(plot) + 1.2))
        ax.barh(labels, plot[lead], color=np.where(plot[lead] > 0, "#b2182b", "#2166ac"))
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel(f"{lead.removeprefix('coef_')} coefficient (per SD of PGS)")
        ax.set_title(f"PGS selected by {lead.removeprefix('coef_')}")
        fig.tight_layout()
        path = cfg.work_dir / "figures" / "top_scores.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
