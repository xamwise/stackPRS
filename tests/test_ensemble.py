import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from conftest import simulate
from stackprs import ensemble as E
from stackprs import learners as L


def test_stacker_nonnegative_and_ignores_noise():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.5, 2000)
    good = 1 / (1 + np.exp(-(2 * y - 1 + rng.standard_normal(2000))))
    noise = rng.uniform(0.05, 0.95, 2000)
    anti = 1 - good                                   # would need a negative weight
    s = E.NonNegativeStacker(["good", "noise", "anti"]).fit(np.column_stack([good, noise, anti]), y)
    w = s.weights()
    assert all(v >= 0 for v in w.values())
    assert w["good"] > 0.5 and w["noise"] < 0.1 and w["anti"] < 1e-6
    assert s.shares()["good"] > 0.8


def test_stacker_with_covariates_requires_them():
    rng = np.random.default_rng(1)
    P, y, Z = rng.uniform(0.1, 0.9, (300, 2)), rng.binomial(1, 0.5, 300), rng.standard_normal((300, 2))
    s = E.NonNegativeStacker(["a", "b"]).fit(P, y, Z)
    assert s.predict_proba(P, Z).shape == (300,)
    with pytest.raises(ValueError):
        s.predict_proba(P)


def _factory(name, seed, final=False):
    return {"lasso": L.LassoL1(n_folds=3, n_cs=6, seed=seed),
            "lasso2": L.LassoL1(n_folds=4, n_cs=5, seed=seed + 1)}[name]


def test_nested_cv_and_final_model(tmp_path):
    X, y, _ = simulate(n=700, p=25, seed=5)
    Z = pd.DataFrame({"sex": np.random.default_rng(2).integers(0, 2, len(y))})
    pred, fold_id, weights = E.cross_validate(["lasso", "lasso2"], _factory, X, y, pgs_cols=list(X.columns), Z=Z,
                                             outer_folds=3, test_size=0.2, inner_folds=3, seed=0)
    assert (fold_id >= 0).all() and not pred.isna().any().any()     # every row held out once
    assert set(pred.columns) >= {"ensemble", "ensemble_plus_covariates", "covariates_only", "best_single_pgs"}
    assert roc_auc_score(y, pred["ensemble"]) > roc_auc_score(y, pred["covariates_only"])
    m = E.metrics_table(pred, y, fold_id, n_boot=50)
    assert {"auc", "auc_ci_low", "delta_auc_vs_ref", "brier"} <= set(m.columns)
    assert len(weights) == 3

    fit = E.fit_stack(["lasso", "lasso2"], _factory, X, y, inner_folds=3, seed=0, Z=Z)
    model = E.StackedModel(names=["lasso", "lasso2"], features=list(X.columns), covariates=["sex"],
                           learners=fit.learners, stacker=fit.stacker, stacker_cov=fit.stacker_cov,
                           scaling=pd.DataFrame({"mean": 0.0, "sd": 1.0}, index=X.columns))
    model.save(tmp_path / "model.pkl")
    again = E.StackedModel.load(tmp_path / "model.pkl")
    assert np.allclose(again.predict_proba(X), model.predict_proba(X))
    assert np.allclose(again.predict_proba(X, Z), model.predict_proba(X, Z))
    assert again.standardize(X * 1.0).shape == X.shape


def test_holdout_mode_leaves_training_rows_unscored():
    X, y, _ = simulate(n=400, p=10, seed=6)
    pred, fold_id, _ = E.cross_validate(["lasso"], _factory, X, y, pgs_cols=list(X.columns), Z=None,
                                        outer_folds=1, test_size=0.25, inner_folds=3, seed=0)
    assert (fold_id == 0).sum() == 100 and pred.loc[fold_id == -1].isna().all().all()
