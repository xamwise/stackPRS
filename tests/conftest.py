"""Shared test helpers. All tests run offline and in seconds; learners whose
software is missing (R uniLasso, Python uniLasso, AutoGluon) are skipped."""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from stackprs import config  # noqa: E402


def load_script(name: str):
    """Import a numbered pipeline script (e.g. '02_filter_and_chunk') as a module."""
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def make_cfg(tmp_path, monkeypatch):
    """Config pointing at a temp work dir; config.load() returns it inside scripts."""
    def _make(**overrides):
        raw = {
            "paths": {"plink_prefix": str(tmp_path / "geno"), "pheno": str(tmp_path / "pheno.txt"),
                      "covariate": str(tmp_path / "cov.txt"), "work_dir": str(tmp_path / "data")},
            "target": {"build": "GRCh37", "sampleset": "COH", "format": "bfile", "pheno_col": "phenotype"},
            "catalog": {"api_base": "https://example.invalid/rest", "page_size": 250},
            "filters": {}, "chunk": {"size": 2}, "run": {"min_overlap": 0.75},
        }
        for key, value in overrides.items():
            raw[key] = value
        cfg = config.Config(raw=raw)
        monkeypatch.setattr(config, "load", lambda *a, **k: cfg)
        return cfg
    return _make


def simulate(n=600, p=40, n_signal=5, seed=0, effect=(0.4, 0.8)):
    """Case/control data where the first `n_signal` features carry signal.

    ``effect=(0.03, 0.12)`` mimics real PRS: many weak effects, so the CV-optimal
    lambda sits mid-path (the regime where uniLasso's extract_cv bug shows).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:n_signal] = rng.choice([-1, 1], n_signal) * rng.uniform(*effect, n_signal)
    y = rng.binomial(1, 1 / (1 + np.exp(-(X @ beta)))).astype(int)
    cols = [f"PGS{i:06d}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), y, beta
