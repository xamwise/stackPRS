"""Stacked ensemble on top of pretrained PRS ("fine-tuning" for one disease).

Level 0  pretrained PRS from the PGS Catalog (the features)
Level 1  base learners (:mod:`stackprs.learners`): uniLasso (Python), uniLasso (R),
         AutoGluon, lasso
Level 2  :class:`NonNegativeStacker`: logistic regression on the base learners'
         logits with non-negative weights, fitted on *out-of-fold* predictions

Evaluation is nested. Each outer split holds out test individuals; inside the
outer training set, inner K-fold produces out-of-fold base-learner predictions
that train the stacker, and the base learners are then refit on the whole outer
training set. No label of a test individual is used to fit anything that
predicts it.
"""
from __future__ import annotations

import pickle
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .learners import BestSinglePGS, CovariatesOnly


def _logit(p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


class NonNegativeStacker:
    """``logit P(case) = a + sum_k w_k * logit(p_k) [+ sum_j c_j * z_j]``, ``w_k >= 0``.

    Non-negative weights are the standard stacking constraint (Breiman 1996):
    they keep the combination interpretable and stable when base learners are
    highly correlated (e.g. the two uniLasso implementations). Optional
    covariates ``z`` enter unconstrained (standardised internally). A small
    ridge penalty on the slopes breaks ties between near-duplicate learners.
    """

    def __init__(self, names: list[str], l2: float = 1e-3):
        self.names, self.l2 = list(names), l2

    def _design(self, P, Z=None, fit: bool = False) -> np.ndarray:
        F = _logit(P)
        cols = [np.ones(len(F)), F]
        if Z is not None:
            Z = np.asarray(Z, dtype=float)
            if fit:
                self.z_mu_, self.z_sd_ = Z.mean(axis=0), Z.std(axis=0)
                self.z_sd_[self.z_sd_ == 0] = 1.0
            cols.append((Z - self.z_mu_) / self.z_sd_)
        return np.column_stack(cols)

    def fit(self, P, y, Z=None) -> "NonNegativeStacker":
        y = np.asarray(y, dtype=float)
        A = self._design(P, Z, fit=True)
        K, q = len(self.names), A.shape[1]
        self.has_covariates_ = Z is not None

        def objective(b):
            eta = A @ b
            loss = np.mean(np.logaddexp(0.0, eta) - y * eta) + self.l2 * np.sum(b[1:] ** 2)
            grad = A.T @ (expit(eta) - y) / len(y)
            grad[1:] += 2 * self.l2 * b[1:]
            return loss, grad

        x0 = np.zeros(q)
        x0[1:1 + K] = 1.0 / K
        bounds = [(None, None)] + [(0.0, None)] * K + [(None, None)] * (q - 1 - K)
        res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds)
        if not res.success:   # L-BFGS-B often flags line-search trouble at a good optimum
            warnings.warn(f"Stacker optimisation: {res.message} (loss {res.fun:.6f})", RuntimeWarning)
        self.coef_ = res.x
        self.logit_sd_ = _logit(P).std(axis=0)
        return self

    def predict_proba(self, P, Z=None) -> np.ndarray:
        if self.has_covariates_ != (Z is not None):
            raise ValueError("Covariates must be supplied iff the stacker was fitted with them")
        return expit(self._design(P, Z) @ self.coef_)

    def weights(self) -> dict[str, float]:
        """Raw non-negative weight per base learner."""
        return dict(zip(self.names, self.coef_[1:1 + len(self.names)]))

    def shares(self) -> dict[str, float]:
        """Share of the ensemble's genetic signal per learner (weight x SD of its logit)."""
        contrib = self.coef_[1:1 + len(self.names)] * self.logit_sd_
        total = contrib.sum()
        return dict(zip(self.names, contrib / total if total > 0 else contrib))


# --------------------------------------------------------------------------- #
# Stacking
# --------------------------------------------------------------------------- #
Factory = Callable[..., object]   # factory(name, seed, final=False) -> unfitted learner


@dataclass
class StackFit:
    learners: dict[str, object]
    stacker: NonNegativeStacker
    stacker_cov: NonNegativeStacker | None
    oof: np.ndarray                    # inner out-of-fold P(case), shape [n, n_learners]


def _log(log, msg):
    if log:
        log(msg)


def fit_stack(names: list[str], factory: Factory, X: pd.DataFrame, y, *, inner_folds: int,
              seed: int, Z: pd.DataFrame | None = None, final: bool = False,
              log=None, tag: str = "") -> StackFit:
    """Fit base learners and the stacker on (X, y) with out-of-fold stacking."""
    y = np.asarray(y).astype(int)
    oof = np.full((len(y), len(names)), np.nan)
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(inner.split(np.zeros(len(y)), y)):
        for j, name in enumerate(names):
            t0 = time.time()
            learner = factory(name, seed + k)
            learner.fit(X.iloc[tr], y[tr])
            oof[va, j] = learner.predict_proba(X.iloc[va])
            learner.cleanup()
            _log(log, f"  {tag}inner {k + 1}/{inner_folds}  {name:<12} {time.time() - t0:7.1f}s")
    learners = {}
    for name in names:
        t0 = time.time()
        learners[name] = factory(name, seed, final=final).fit(X, y)
        _log(log, f"  {tag}refit          {name:<12} {time.time() - t0:7.1f}s")
    stacker = NonNegativeStacker(names).fit(oof, y)
    stacker_cov = NonNegativeStacker(names).fit(oof, y, Z) if Z is not None else None
    return StackFit(learners, stacker, stacker_cov, oof)


def outer_splits(y, outer_folds: int, test_size: float, seed: int):
    """Outer evaluation splits: K-fold (nested CV) or one stratified holdout."""
    y = np.asarray(y)
    if outer_folds >= 2:
        return list(StratifiedKFold(outer_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return list(StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                .split(np.zeros(len(y)), y))


def cross_validate(names: list[str], factory: Factory, X: pd.DataFrame, y, *,
                   pgs_cols: list[str], Z: pd.DataFrame | None, outer_folds: int,
                   test_size: float, inner_folds: int, seed: int, log=None, on_fold=None):
    """Nested evaluation of base learners, the stacked ensemble and baselines.

    ``on_fold(fold, stack_fit, X_test, y_test)`` is called before each fold's
    models are discarded (e.g. for permutation importance on held-out data).
    Returns ``(predictions, fold_id, stacker_weights)``; rows never held out
    (holdout mode) have ``fold_id == -1`` and NaN predictions.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    models = list(names) + ["ensemble"]
    if Z is not None:
        models += ["ensemble_plus_covariates", "covariates_only"]
    models += ["best_single_pgs"]
    pred = pd.DataFrame(np.nan, index=np.arange(n), columns=models)
    fold_id = np.full(n, -1)
    weights = []
    splits = outer_splits(y, outer_folds, test_size, seed)
    for f, (tr, te) in enumerate(splits):
        tag = f"outer {f + 1}/{len(splits)} · "
        _log(log, f"{tag}train={len(tr)} test={len(te)}")
        Ztr = Z.iloc[tr] if Z is not None else None
        fit = fit_stack(names, factory, X.iloc[tr], y[tr], inner_folds=inner_folds,
                        seed=seed + 1000 * (f + 1), Z=Ztr, log=log, tag=tag)
        P = np.column_stack([fit.learners[m].predict_proba(X.iloc[te]) for m in names])
        pred.loc[te, names] = P
        pred.loc[te, "ensemble"] = fit.stacker.predict_proba(P)
        if Z is not None:
            pred.loc[te, "ensemble_plus_covariates"] = fit.stacker_cov.predict_proba(P, Z.iloc[te])
            pred.loc[te, "covariates_only"] = CovariatesOnly().fit(Ztr, y[tr]).predict_proba(Z.iloc[te])
        best = BestSinglePGS().fit(X[pgs_cols].iloc[tr], y[tr])
        pred.loc[te, "best_single_pgs"] = best.predict_proba(X[pgs_cols].iloc[te])
        if on_fold is not None:
            on_fold(f, fit, X.iloc[te], y[te])
        for learner in fit.learners.values():
            learner.cleanup()
        fold_id[te] = f
        weights.append({"fold": f, "best_single_pgs": best.feature_,
                        **{f"w_{k}": v for k, v in fit.stacker.weights().items()},
                        **{f"share_{k}": v for k, v in fit.stacker.shares().items()}})
        _log(log, f"{tag}AUC " + "  ".join(f"{m}={roc_auc_score(y[te], pred.loc[te, m]):.3f}" for m in models))
    return pred, fold_id, pd.DataFrame(weights)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics_table(pred: pd.DataFrame, y, fold_id, *, reference: str = "ensemble",
                  n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Discrimination + calibration per model, with bootstrap CIs.

    ``delta_auc_vs_ref`` is AUC(reference) - AUC(model) from a paired bootstrap
    (same resampled individuals for both), so its CI answers "is the ensemble
    better than this model?".
    """
    y = np.asarray(y).astype(int)
    held = fold_id >= 0
    yy, P = y[held], pred.loc[held]
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(yy), len(yy)) for _ in range(n_boot)]
    boots = [b for b in boots if 0 < yy[b].sum() < len(b)]
    boot_auc = pd.DataFrame({m: [roc_auc_score(yy[b], P[m].to_numpy()[b]) for b in boots] for m in P.columns})
    auc = {m: roc_auc_score(yy, P[m]) for m in P.columns}
    rows = []
    for m in P.columns:
        p = P[m].to_numpy()
        per_fold = [roc_auc_score(y[fold_id == f], pred.loc[fold_id == f, m])
                    for f in np.unique(fold_id[held])]
        row = {
            "model": m, "n": int(held.sum()),
            "auc": auc[m],
            "auc_ci_low": boot_auc[m].quantile(0.025), "auc_ci_high": boot_auc[m].quantile(0.975),
            "auc_fold_mean": float(np.mean(per_fold)), "auc_fold_sd": float(np.std(per_fold)),
            "auprc": average_precision_score(yy, p),
            "brier": float(np.mean((p - yy) ** 2)),
            "log_loss": log_loss(yy, np.clip(p, 1e-6, 1 - 1e-6)),
        }
        if reference in P.columns and m != reference:
            d = boot_auc[reference] - boot_auc[m]
            row.update(delta_auc_vs_ref=auc[reference] - auc[m],
                       delta_ci_low=d.quantile(0.025), delta_ci_high=d.quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Final model
# --------------------------------------------------------------------------- #
@dataclass
class StackedModel:
    """The fitted disease-specific ensemble, applicable to new individuals.

    ``scaling`` holds the training cohort's mean/sd of each raw PGS so new raw
    pgsc_calc scores are standardised exactly like the training data.
    """
    names: list[str]
    features: list[str]
    covariates: list[str]
    learners: dict[str, object]
    stacker: NonNegativeStacker
    stacker_cov: NonNegativeStacker | None = None
    scaling: pd.DataFrame | None = None
    info: dict = field(default_factory=dict)

    def standardize(self, raw: pd.DataFrame, Z: pd.DataFrame | None = None) -> pd.DataFrame:
        """Transform raw pgsc_calc scores exactly like step 05 did for training.

        Missing scores are imputed at the training mean. If the PGS were
        residualised on covariates (``dataset.residualize_on``), ``Z`` must hold
        those covariates.
        """
        if self.scaling is None:
            raise ValueError("No scaling stored with this model")
        s = self.scaling
        pgs = [c for c in self.features if c in s.index]
        out = ((raw.reindex(columns=pgs) - s.loc[pgs, "mean"]) / s.loc[pgs, "sd"]).fillna(0.0)
        if "resid_intercept" in s.columns:
            adj = [c.removeprefix("resid_") for c in s.columns
                   if c.startswith("resid_") and c not in ("resid_intercept", "resid_sd")]
            if Z is None:
                raise ValueError(f"The PGS were residualised on {adj}; pass them as Z")
            A = np.column_stack([np.ones(len(out)), Z[adj].to_numpy(dtype=float)])
            B = s.loc[pgs, ["resid_intercept"] + [f"resid_{c}" for c in adj]].to_numpy().T
            out = (out - A @ B) / s.loc[pgs, "resid_sd"]
        return out

    def base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.learners[m].predict_proba(X) for m in self.names])

    def predict_proba(self, X: pd.DataFrame, Z: pd.DataFrame | None = None) -> np.ndarray:
        """Ensemble P(case); with ``Z`` (covariates) uses the covariate-adjusted stacker."""
        P = self.base_predictions(X)
        if Z is None:
            return self.stacker.predict_proba(P)
        if self.stacker_cov is None:
            raise ValueError("This model was fitted without covariates")
        return self.stacker_cov.predict_proba(P, Z[self.covariates])

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "StackedModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)
