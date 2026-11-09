"""Step 05 end-to-end, and train/inference consistency of the feature transform."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import load_script
from stackprs import data
from stackprs.ensemble import StackedModel


@pytest.fixture
def cohort(make_cfg, tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    iid = [str(1000 + i) for i in range(n)]
    y = rng.integers(0, 2, n)
    pc1 = rng.standard_normal(n) + 1.5 * y                # PC that (artefactually) tracks case status
    raw = pd.DataFrame({"IID": iid,
                        "PGS000001": 10 + 3 * pc1 + rng.standard_normal(n),   # mostly PC structure
                        "PGS000002": -2 + 0.5 * rng.standard_normal(n) + 0.3 * y})
    cfg = make_cfg(dataset={"min_complete": 0.9, "residualize_on": ["PC1"]})
    data.write_frame(raw, cfg.work_dir / "pgs_feature_matrix")
    pd.DataFrame({"FID": iid, "IID": iid, "phenotype": y}).to_csv(tmp_path / "pheno.txt", sep=" ", index=False)
    pd.DataFrame({"FID": iid, "IID": iid, "Sex": rng.integers(1, 3, n), "PC1": pc1}) \
        .to_csv(tmp_path / "cov.txt", sep=" ", index=False)
    return cfg, raw


def test_residualisation_and_confounding_warning(cohort, capsys, monkeypatch):
    cfg, raw = cohort
    monkeypatch.setattr("sys.argv", ["05_build_dataset.py"])
    load_script("05_build_dataset").main()
    out = capsys.readouterr().out
    assert "WARNING: covariate PC1 alone separates cases from controls" in out

    ds = data.read_frame(cfg.work_dir / "model_dataset")
    cols = json.loads((cfg.work_dir / "dataset_columns.json").read_text())
    assert cols["residualized_on"] == ["PC1"]
    feats = ds[cols["pgs"]]
    assert np.allclose(feats.std(ddof=0), 1.0)
    assert abs(np.corrcoef(feats["PGS000001"], ds["PC1"])[0, 1]) < 1e-8     # PC structure removed

    # a saved model must transform new raw scores exactly like training did
    scaling = pd.read_csv(cfg.work_dir / "feature_scaling.csv", index_col="pgs_id")
    model = StackedModel(names=[], features=cols["pgs"], covariates=cols["covariates"],
                         learners={}, stacker=None, scaling=scaling)
    Z = ds[["PC1"]]
    again = model.standardize(raw.drop(columns="IID"), Z=Z)
    assert np.allclose(again.to_numpy(), feats.to_numpy())
    with pytest.raises(ValueError, match="residualised"):
        model.standardize(raw.drop(columns="IID"))
