"""Base learners fitted on top of the pretrained (PGS Catalog) PRS.

Every learner has the same small interface used by :mod:`stackprs.ensemble`::

    learner.fit(X: DataFrame, y: 0/1 array) -> learner
    learner.predict_proba(X) -> P(case) as a 1-D array
    learner.coefficients() -> Series of per-feature coefficients, or None
    learner.cleanup()      -> free disk/memory of throw-away CV fits

Learners
--------
unilasso_py  uniLasso, Python implementation (github.com/sophial05/uni-lasso)
unilasso_r   uniLasso, R implementation (CRAN ``uniLasso``, via Rscript)
autogluon    AutoGluon TabularPredictor (AutoML: boosted trees, NNs, own ensembling)
lasso        plain L1 logistic regression — the reference uniLasso is benchmarked against

Baselines (not stacked): ``covariates_only`` and ``best_single_pgs``.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

R_SCRIPT = pathlib.Path(__file__).resolve().parent / "r" / "unilasso_fit.R"


@contextlib.contextmanager
def silence_output():
    """Silence stdout/stderr at the file-descriptor level.

    adelie (the uniLasso Python backend) draws C++ progress bars that bypass
    ``sys.stderr``, so ``contextlib.redirect_stderr`` is not enough.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def column_aucs(X: pd.DataFrame, y) -> pd.Series:
    """Univariate ROC AUC of every column (vectorised Mann-Whitney U)."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    ranks = rankdata(np.asarray(X, dtype=float), axis=0)
    auc = (ranks[y].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return pd.Series(auc, index=X.columns)


class _LinearMixin:
    """Shared prediction for learners reduced to intercept + coefficients."""
    features_: list[str]
    coef_: np.ndarray
    intercept_: float

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X[self.features_], dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return expit(self.decision_function(X))

    def coefficients(self) -> pd.Series:
        return pd.Series(self.coef_, index=self.features_, name=self.name)

    def cleanup(self) -> None:
        pass


class UniLassoPy(_LinearMixin):
    """uniLasso via the Python package (``cv_unilasso``).

    Two package pitfalls are handled here:

    * ``predict`` returns the *linear predictor* even for the binomial family,
      so we keep the coefficients and apply the logistic link ourselves.
    * ``extract_cv`` is wrong (at commit c184540): after cross-validation,
      adelie refits along a *new* path that ends at the best lambda, but
      ``extract_cv`` takes row ``best_idx`` of that refit path — an index into
      the *original* CV path. The returned model is therefore fitted at a much
      larger lambda (8x in our data: 3 PGS selected instead of 16) while
      ``best_lmda`` still reports the right value. We take the refit row whose
      lambda equals ``best_lmda`` instead.
    """
    name = "unilasso_py"

    def __init__(self, n_folds: int = 10, seed: int = 0):
        self.n_folds, self.seed = n_folds, seed

    def fit(self, X: pd.DataFrame, y) -> "UniLassoPy":
        from unilasso import cv_unilasso
        with silence_output():
            cv = cv_unilasso(np.asarray(X, dtype=float), np.asarray(y, dtype=float),
                             family="binomial", n_folds=self.n_folds, seed=self.seed)
        best = float(cv.best_lmda)
        refit_lmdas = np.asarray(cv.lasso_model.lmdas, dtype=float)
        row = int(np.argmin(np.abs(refit_lmdas - best)))
        if not np.isclose(refit_lmdas[row], best, rtol=1e-6):
            raise RuntimeError(f"uniLasso refit path does not contain best lambda {best:.4g}")
        coefs = np.atleast_2d(np.asarray(cv.coefs, dtype=float))
        self.features_ = list(X.columns)
        self.coef_ = coefs[row].ravel()
        self.intercept_ = float(np.atleast_1d(np.asarray(cv.intercept, dtype=float))[row])
        self.lambda_ = best
        return self


class UniLassoR(_LinearMixin):
    """uniLasso via the CRAN R package, run with ``Rscript``.

    Data go to R as raw little-endian float64 (column-major), so no extra R
    packages are needed; R returns the coefficients at the chosen lambda, and
    prediction happens here (identical to ``predict(..., type="response")``).
    """
    name = "unilasso_r"

    def __init__(self, n_folds: int = 10, seed: int = 0, lambda_rule: str = "lambda.min",
                 rscript: str = "Rscript"):
        if lambda_rule not in ("lambda.min", "lambda.1se"):
            raise ValueError("lambda_rule must be 'lambda.min' or 'lambda.1se'")
        self.n_folds, self.seed, self.lambda_rule, self.rscript = n_folds, seed, lambda_rule, rscript

    def fit(self, X: pd.DataFrame, y) -> "UniLassoR":
        Xa = np.asarray(X, dtype=float)
        n, p = Xa.shape
        with tempfile.TemporaryDirectory(prefix="unilasso_r_") as wd:
            wd = pathlib.Path(wd)
            np.ascontiguousarray(Xa.T).astype("<f8").tofile(wd / "X.bin")   # column-major for R
            np.asarray(y, dtype="<f8").tofile(wd / "y.bin")
            cmd = [self.rscript, str(R_SCRIPT), str(wd), str(n), str(p),
                   str(self.n_folds), str(self.seed), self.lambda_rule]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"R uniLasso failed (rc={res.returncode}):\n{res.stderr[-3000:]}")
            coef = np.fromfile(wd / "coef.bin", dtype="<f8")
            info = json.loads((wd / "fit.json").read_text())
        if coef.shape != (p + 1,):
            raise RuntimeError(f"R returned {coef.shape[0]} coefficients, expected {p + 1}")
        self.features_ = list(X.columns)
        self.intercept_, self.coef_ = float(coef[0]), coef[1:]
        self.lambda_ = float(info["lambda"])
        return self


def check_r_unilasso(rscript: str = "Rscript") -> str:
    """Return the installed R uniLasso version, or raise with install instructions."""
    if shutil.which(rscript) is None and not pathlib.Path(rscript).exists():
        raise RuntimeError(f"{rscript!r} not found: install R (>= 3.6) or set ensemble.unilasso.rscript, "
                           "or drop 'unilasso_r' from ensemble.learners")
    res = subprocess.run([rscript, "-e", 'cat(as.character(packageVersion("uniLasso")))'],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("R package 'uniLasso' is not installed. In R run: "
                           "install.packages('uniLasso')  (or `make r-deps`)")
    return res.stdout.strip()


class LassoL1(_LinearMixin):
    """Standard L1-penalised logistic regression (lambda by internal CV)."""
    name = "lasso"

    def __init__(self, n_folds: int = 10, n_cs: int = 20, seed: int = 0):
        self.n_folds, self.n_cs, self.seed = n_folds, n_cs, seed

    def fit(self, X: pd.DataFrame, y) -> "LassoL1":
        import sklearn
        from sklearn.linear_model import LogisticRegressionCV
        major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
        # scikit-learn 1.8 deprecated `penalty`; before 1.8 `l1_ratios` is ignored
        # unless penalty="elasticnet", which would silently fit a ridge model.
        l1 = {"l1_ratios": (1.0,)} if (major, minor) >= (1, 8) else {"penalty": "l1"}
        Xa = np.asarray(X, dtype=float)
        mu, sd = Xa.mean(axis=0), Xa.std(axis=0)
        sd[sd == 0] = 1.0
        m = LogisticRegressionCV(Cs=self.n_cs, cv=self.n_folds, solver="saga", scoring="neg_log_loss",
                                 max_iter=5000, n_jobs=-1, random_state=self.seed, **l1)
        m.fit((Xa - mu) / sd, np.asarray(y).astype(int))
        # fold the standardisation into the coefficients -> predictions on raw X
        # (np.ravel: scikit-learn 1.10 changes the shapes of LogisticRegressionCV attributes)
        w, b = np.ravel(m.coef_), float(np.ravel(m.intercept_)[0])
        self.features_ = list(X.columns)
        self.coef_ = w / sd
        self.intercept_ = b - float(np.sum(w * mu / sd))
        self.C_ = float(np.ravel(m.C_)[0])
        return self


class AutoGluonLearner:
    """AutoGluon ``TabularPredictor`` for the binary outcome.

    Throw-away CV fits are deleted by :meth:`cleanup`; the final model is kept
    under ``path`` and reloaded lazily after unpickling.
    """
    name = "autogluon"
    _LABEL = "__case__"

    def __init__(self, presets="medium_quality", time_limit: float | None = 300,
                 eval_metric: str = "roc_auc", num_cpus="auto", hyperparameters=None,
                 path: str | pathlib.Path | None = None, verbosity: int = 0):
        self.presets, self.time_limit, self.eval_metric = presets, time_limit, eval_metric
        self.num_cpus, self.hyperparameters, self.verbosity = num_cpus, hyperparameters, verbosity
        self.path = str(path) if path else None
        self._predictor = None

    def fit(self, X: pd.DataFrame, y) -> "AutoGluonLearner":
        from autogluon.tabular import TabularPredictor
        if self._LABEL in X.columns:
            raise ValueError(f"Feature named {self._LABEL!r} collides with the label column")
        self.path_ = self.path or tempfile.mkdtemp(prefix="autogluon_")
        self.features_ = list(X.columns)
        train = X.assign(**{self._LABEL: np.asarray(y).astype(int)})
        self._predictor = TabularPredictor(
            label=self._LABEL, problem_type="binary", eval_metric=self.eval_metric,
            path=self.path_, verbosity=self.verbosity,
        ).fit(train, presets=self.presets, time_limit=self.time_limit,
              num_cpus=self.num_cpus, hyperparameters=self.hyperparameters)
        return self

    @property
    def predictor(self):
        if self._predictor is None:
            from autogluon.tabular import TabularPredictor
            self._predictor = TabularPredictor.load(self.path_, verbosity=0)
        return self._predictor

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.predictor.predict_proba(X[self.features_], as_multiclass=False), dtype=float)

    def coefficients(self) -> None:
        return None

    def cleanup(self) -> None:
        self._predictor = None
        if self.path is None and getattr(self, "path_", None):
            shutil.rmtree(self.path_, ignore_errors=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_predictor"] = None          # the predictor lives on disk at path_
        return state


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
class CovariatesOnly:
    """Logistic regression on covariates only (e.g. sex + genetic PCs)."""
    name = "covariates_only"

    def fit(self, X: pd.DataFrame, y) -> "CovariatesOnly":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.features_ = list(X.columns)
        self.model_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X, np.asarray(y).astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.features_])[:, 1]


class BestSinglePGS:
    """The single catalog PRS with the highest training AUC, as-is.

    Answers "does the ensemble beat the best off-the-shelf score?". Selection
    uses training data only; direction is allowed to be protective.
    """
    name = "best_single_pgs"

    def fit(self, X: pd.DataFrame, y) -> "BestSinglePGS":
        from sklearn.linear_model import LogisticRegression
        aucs = column_aucs(X, y)
        self.feature_ = str((aucs - 0.5).abs().idxmax())
        self.train_auc_ = float(aucs[self.feature_])
        self.model_ = LogisticRegression(max_iter=1000).fit(X[[self.feature_]], np.asarray(y).astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[[self.feature_]])[:, 1]


def make_learner(name: str, cfg, *, seed: int = 0, path: str | pathlib.Path | None = None):
    """Construct an unfitted base learner from the ``ensemble`` config section."""
    if name == "unilasso_py":
        return UniLassoPy(n_folds=int(cfg.get("ensemble.unilasso.n_folds", 10)), seed=seed)
    if name == "unilasso_r":
        return UniLassoR(n_folds=int(cfg.get("ensemble.unilasso.n_folds", 10)), seed=seed,
                         lambda_rule=cfg.get("ensemble.unilasso.lambda_rule", "lambda.min"),
                         rscript=cfg.get("ensemble.unilasso.rscript", "Rscript"))
    if name == "lasso":
        return LassoL1(n_folds=int(cfg.get("ensemble.lasso.n_folds", 10)),
                       n_cs=int(cfg.get("ensemble.lasso.n_cs", 20)), seed=seed)
    if name == "autogluon":
        return AutoGluonLearner(presets=cfg.get("ensemble.autogluon.presets", "medium_quality"),
                                time_limit=cfg.get("ensemble.autogluon.time_limit", 300),
                                eval_metric=cfg.get("ensemble.autogluon.eval_metric", "roc_auc"),
                                num_cpus=cfg.get("ensemble.autogluon.num_cpus", "auto"),
                                hyperparameters=cfg.get("ensemble.autogluon.hyperparameters"),
                                path=path)
    raise ValueError(f"Unknown learner {name!r}; choose from unilasso_py, unilasso_r, autogluon, lasso")
