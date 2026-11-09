#!/usr/bin/env python3
"""Step 6 — fit the disease-specific ensemble on top of the pretrained PRS.

Base learners (ensemble.learners):
  unilasso_py  uniLasso, Python (github.com/sophial05/uni-lasso)
  unilasso_r   uniLasso, R (CRAN 'uniLasso', via Rscript)
  autogluon    AutoGluon TabularPredictor (AutoML)
  lasso        L1 logistic regression (the reference uniLasso improves on)
Stacker: non-negative logistic regression on the learners' out-of-fold logits.
Baselines: covariates only; best single catalog PRS.

1. Nested evaluation (ensemble.outer_folds; 1 = single stratified holdout)
   -> data/ensemble/cv_metrics.csv, cv_predictions.csv, cv_stacker_weights.csv,
      cv_autogluon_importance.csv (permutation importance on held-out folds)
2. Final fit on all individuals
   -> data/ensemble/final/model.pkl (+ autogluon/), coefficients.csv, stacker.json

  python scripts/06_fit_ensemble.py            # full run (see README for runtime)
  python scripts/06_fit_ensemble.py --quick    # holdout, 3 inner folds, AutoGluon 60 s
"""
import _bootstrap  # noqa: F401

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from stackprs import config, data
from stackprs.ensemble import StackedModel, cross_validate, fit_stack, metrics_table
from stackprs.learners import check_r_unilasso, column_aucs, make_learner

LEARNERS = ("unilasso_py", "unilasso_r", "autogluon", "lasso")
LINEAR = ("unilasso_py", "unilasso_r", "lasso")


def preflight(names: list[str], cfg) -> dict:
    """Fail fast with install hints; return versions for the run record."""
    versions = {"python": platform.python_version()}
    import sklearn
    versions["scikit-learn"] = sklearn.__version__
    if "unilasso_py" in names:
        try:
            import unilasso
        except ImportError:
            sys.exit("Python uniLasso missing: pip install "
                     "'unilasso @ git+https://github.com/sophial05/uni-lasso.git@c184540'")
        versions["unilasso_py"] = unilasso.__version__
    if "unilasso_r" in names:
        try:
            versions["unilasso_r"] = check_r_unilasso(cfg.get("ensemble.unilasso.rscript", "Rscript"))
        except RuntimeError as e:
            sys.exit(str(e))
    if "autogluon" in names:
        try:
            import autogluon.tabular as agt
        except ImportError:
            sys.exit("AutoGluon missing: pip install 'autogluon.tabular[all]==1.6.3'")
        versions["autogluon"] = agt.__version__
    return versions


def main() -> None:
    cfg = config.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fast smoke run: holdout, 3 inner folds, AutoGluon 60 s")
    ap.add_argument("--skip-cv", action="store_true", help="only fit the final model")
    ap.add_argument("--learners", help="comma-separated override of ensemble.learners")
    args = ap.parse_args()

    names = args.learners.split(",") if args.learners else list(cfg.get("ensemble.learners", LEARNERS))
    unknown = sorted(set(names) - set(LEARNERS))
    if unknown or not names:
        sys.exit(f"Unknown/empty learners {unknown}; choose from {LEARNERS}")
    outer = int(cfg.get("ensemble.outer_folds", 5))
    inner = int(cfg.get("ensemble.inner_folds", 5))
    test_size = float(cfg.get("ensemble.test_size", 0.2))
    seed = int(cfg.get("ensemble.seed", 42))
    ag_time = cfg.get("ensemble.autogluon.time_limit", 300)
    if args.quick:
        outer, inner, ag_time = 1, 3, 60
    versions = preflight(names, cfg)

    ds = data.read_frame(cfg.work_dir / "model_dataset")
    cols = json.loads((cfg.work_dir / "dataset_columns.json").read_text())
    pgs, covs = cols["pgs"], cols["covariates"]
    use_cov = bool(cfg.get("ensemble.use_covariates", False))
    features = pgs + (covs if use_cov else [])
    X, y = ds[features], ds[cols["phenotype"]].to_numpy(dtype=int)
    Z = ds[covs] if covs else None

    out_dir, final_dir = cfg.ensemble_dir, cfg.ensemble_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    logfile = open(out_dir / "fit.log", "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logfile.write(line + "\n")
        logfile.flush()

    def factory(name, seed, final=False):
        learner = make_learner(name, cfg, seed=seed,
                               path=final_dir / "autogluon" if final and name == "autogluon" else None)
        if name == "autogluon":
            learner.time_limit = ag_time
        return learner

    log(f"N={len(y)} ({int(y.sum())} cases) | features={len(features)} (PGS={len(pgs)}, "
        f"covariates in learners={use_cov}) | learners={names} | outer={outer} inner={inner} | {versions}")

    # ---- 1. nested evaluation -------------------------------------------------
    if not args.skip_cv:
        importance = []
        want_fi = "autogluon" in names and cfg.get("importance.autogluon_permutation", True)

        def on_fold(f, fit, X_te, y_te):
            if not want_fi:
                return
            # candidates: PGS selected by any linear learner in this fold + top univariate AUC
            selected = set()
            for name in LINEAR:
                if name in fit.learners:
                    c = fit.learners[name].coefficients()
                    selected |= set(c.index[c != 0])
            max_n = int(cfg.get("importance.autogluon_max_features", 200))
            by_auc = (column_aucs(X_te[pgs], y_te) - 0.5).abs().sort_values(ascending=False).index
            cand = [c for c in features if c in selected][:max_n]
            cand += [c for c in by_auc if c not in cand][: max(0, max_n - len(cand))]
            t0 = time.time()
            fi = fit.learners["autogluon"].predictor.feature_importance(
                X_te.assign(__case__=y_te), features=cand,
                subsample_size=int(cfg.get("importance.subsample_size", 5000)),
                num_shuffle_sets=int(cfg.get("importance.num_shuffle_sets", 3)),
                time_limit=cfg.get("importance.time_limit", 600), silent=True)
            importance.append(fi.rename_axis("feature").reset_index().assign(fold=f))
            log(f"  outer {f + 1} · AutoGluon permutation importance on held-out fold "
                f"({len(cand)} features, {time.time() - t0:.0f}s)")

        pred, fold_id, weights = cross_validate(
            names, factory, X, y, pgs_cols=pgs, Z=Z, outer_folds=outer, test_size=test_size,
            inner_folds=inner, seed=seed, log=log, on_fold=on_fold)
        pd.concat([pd.DataFrame({"IID": ds["IID"], "y": y, "fold": fold_id}), pred], axis=1) \
            .to_csv(out_dir / "cv_predictions.csv", index=False)
        weights.to_csv(out_dir / "cv_stacker_weights.csv", index=False)
        if importance:
            pd.concat(importance).to_csv(out_dir / "cv_autogluon_importance.csv", index=False)
        metrics = metrics_table(pred, y, fold_id, reference="ensemble",
                                n_boot=int(cfg.get("ensemble.n_bootstrap", 1000)), seed=seed)
        metrics.to_csv(out_dir / "cv_metrics.csv", index=False)
        show = ["model", "auc", "auc_ci_low", "auc_ci_high", "auc_fold_sd", "auprc", "brier",
                "delta_auc_vs_ref", "delta_ci_low", "delta_ci_high"]
        log("Held-out performance (delta = ensemble AUC minus model AUC, paired bootstrap 95% CI):\n"
            + metrics[[c for c in show if c in metrics]].to_string(index=False, float_format="%.3f"))

    # ---- 2. final model on everyone ---------------------------------------------
    if (final_dir / "autogluon").exists():
        shutil.rmtree(final_dir / "autogluon")          # our own artefact from a previous run
    fit = fit_stack(names, factory, X, y, inner_folds=inner, seed=seed, Z=Z, final=True,
                    log=log, tag="final · ")
    scaling = pd.read_csv(cfg.work_dir / "feature_scaling.csv", index_col="pgs_id")
    info = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "versions": versions,
            "n": int(len(y)), "n_cases": int(y.sum()), "learners": names, "inner_folds": inner,
            "use_covariates": use_cov, "seed": seed}
    model = StackedModel(names=names, features=features, covariates=covs, learners=fit.learners,
                         stacker=fit.stacker, stacker_cov=fit.stacker_cov, scaling=scaling, info=info)
    model.save(final_dir / "model.pkl")

    coef = pd.DataFrame({n: fit.learners[n].coefficients() for n in names if n in LINEAR})
    if not coef.empty:
        intercepts = pd.DataFrame({n: [fit.learners[n].intercept_] for n in coef.columns}, index=["(intercept)"])
        pd.concat([intercepts, coef]).rename_axis("feature").to_csv(final_dir / "coefficients.csv")
    stacker = {"intercept": float(fit.stacker.coef_[0]), "weights": fit.stacker.weights(),
               "shares": fit.stacker.shares()}
    if fit.stacker_cov is not None:
        k = len(names)
        stacker["with_covariates"] = {
            "intercept": float(fit.stacker_cov.coef_[0]),
            "weights": dict(zip(names, fit.stacker_cov.coef_[1:1 + k])),
            "covariate_coefs_per_sd": dict(zip(covs, fit.stacker_cov.coef_[1 + k:]))}
    (final_dir / "stacker.json").write_text(json.dumps(stacker, indent=1, default=float))
    pd.DataFrame(fit.oof, columns=names).assign(IID=ds["IID"].to_numpy(), y=y) \
        .to_csv(final_dir / "oof_predictions.csv", index=False)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=1))

    shares = ", ".join(f"{k}={v:.0%}" for k, v in fit.stacker.shares().items())
    log(f"Final model -> {final_dir / 'model.pkl'} | learner shares: {shares}")
    for n in coef.columns:
        log(f"  {n}: {int((coef[n] != 0).sum())} PGS selected")
    logfile.close()


if __name__ == "__main__":
    main()
