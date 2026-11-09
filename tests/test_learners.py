import os
import shutil
import subprocess

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from conftest import simulate
from stackprs import learners as L


def r_unilasso_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    res = subprocess.run(["Rscript", "-e", 'quit(status = !requireNamespace("uniLasso", quietly = TRUE))'],
                         capture_output=True)
    return res.returncode == 0


def test_column_aucs_match_sklearn():
    X, y, _ = simulate(n=300, p=6)
    aucs = L.column_aucs(X, y)
    for c in X.columns:
        assert aucs[c] == pytest.approx(roc_auc_score(y, X[c]))


def test_lasso_predictions_on_raw_scale():
    X, y, beta = simulate(n=800, p=30)
    m = L.LassoL1(n_folds=3, n_cs=8).fit(X, y)
    p = m.predict_proba(X)
    assert p.min() > 0 and p.max() < 1 and roc_auc_score(y, p) > 0.7
    assert set(np.flatnonzero(beta)) <= set(np.flatnonzero(m.coef_))   # true signals kept


def test_best_single_pgs_and_covariates_baselines():
    X, y, beta = simulate(n=800, p=20)
    best = L.BestSinglePGS().fit(X, y)
    assert best.feature_ in {X.columns[i] for i in np.flatnonzero(beta)}
    assert L.CovariatesOnly().fit(X.iloc[:, :3], y).predict_proba(X.iloc[:, :3]).shape == (800,)


def test_unilasso_py_probabilities_and_sign_constraint():
    pytest.importorskip("unilasso")
    X, y, beta = simulate(n=1000, p=40)
    m = L.UniLassoPy(n_folds=5, seed=1).fit(X, y)
    p = m.predict_proba(X)
    assert p.min() > 0 and p.max() < 1          # logistic link applied (package returns log-odds)
    uni = L.column_aucs(X, y) - 0.5             # univariate direction
    sel = np.flatnonzero(m.coef_)
    assert len(sel) > 0
    assert np.all(np.sign(m.coef_[sel]) == np.sign(uni.to_numpy()[sel]))   # uniLasso keeps univariate signs


def test_unilasso_py_returns_the_solution_at_best_lambda():
    """Regression: the package's extract_cv returns a row of the refit path fitted
    at a much larger lambda than best_lmda (3 instead of 16 PGS on UKB_CRC)."""
    ad = pytest.importorskip("adelie")
    from unilasso import fit_loo_univariate_models
    X, y, _ = simulate(n=3000, p=150, n_signal=20, seed=0, effect=(0.03, 0.12))   # extract_cv: 1 PGS vs 12
    m = L.UniLassoPy(n_folds=5, seed=1).fit(X, y)
    F = np.asfortranarray(fit_loo_univariate_models(X.to_numpy(float), y.astype(float), family="binomial")["fit"])
    with L.silence_output():                     # stage 2 solved directly at the reported lambda
        direct = ad.grpnet(X=F, glm=ad.glm.binomial(y.astype(float)), groups=None, intercept=True,
                           lmda_path=np.array([m.lambda_]), tol=1e-7,
                           constraints=[ad.constraint.lower(b=np.zeros(1)) for _ in range(F.shape[1])])
    sel_direct = set(np.flatnonzero(direct.betas.toarray()[-1]))
    sel_wrapper = set(np.flatnonzero(m.coef_))
    assert len(sel_wrapper) >= 5
    assert len(sel_direct & sel_wrapper) / len(sel_direct | sel_wrapper) > 0.9


@pytest.mark.skipif(not r_unilasso_available(), reason="R + CRAN uniLasso not installed")
def test_unilasso_r_agrees_with_python():
    pytest.importorskip("unilasso")
    X, y, _ = simulate(n=3000, p=150, n_signal=20, seed=1, effect=(0.03, 0.12))
    r = L.UniLassoR(n_folds=5, seed=1).fit(X, y)
    py = L.UniLassoPy(n_folds=5, seed=1).fit(X, y)
    sr, sp = set(np.flatnonzero(r.coef_)), set(np.flatnonzero(py.coef_))
    assert len(sr & sp) / len(sr | sp) > 0.7             # same method -> (nearly) same selection
    assert np.corrcoef(r.predict_proba(X), py.predict_proba(X))[0, 1] > 0.95


def test_r_learner_reports_missing_package():
    with pytest.raises(RuntimeError, match="not found"):
        L.check_r_unilasso("/nonexistent/Rscript")


@pytest.mark.skipif(not os.environ.get("STACKPRS_TEST_AUTOGLUON"), reason="set STACKPRS_TEST_AUTOGLUON=1 (slow)")
def test_autogluon_learner(tmp_path):
    pytest.importorskip("autogluon.tabular")
    X, y, _ = simulate(n=600, p=10)
    m = L.AutoGluonLearner(time_limit=30, hyperparameters={"GBM": {}}).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (600,) and roc_auc_score(y, p) > 0.6
    path = m.path_
    m.cleanup()
    assert not os.path.exists(path)
