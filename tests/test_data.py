import io

import numpy as np
import pandas as pd
import pytest

from stackprs import data


def write_long(path, rows, extra_cols=True):
    """pgsc_calc v2 aggregated_scores.txt.gz: tab-separated, long format."""
    df = pd.DataFrame(rows, columns=["sampleset", "IID", "PGS", "SUM"])
    if extra_cols:
        df["DENOM"] = 10
        df["AVG"] = df["SUM"] / df["DENOM"]
    df.to_csv(path, sep="\t", index=False)
    return path


def test_accession():
    assert data.pgs_accession("PGS000018_hmPOS_GRCh37") == "PGS000018"
    assert data.pgs_accession("my_custom_score") == "my_custom_score"


def test_long_format_hmpos_ids_and_reference_rows(tmp_path):
    p = write_long(tmp_path / "a.txt.gz", [
        ("COH", "1001", "PGS000001_hmPOS_GRCh37", 1.0), ("COH", "1002", "PGS000001_hmPOS_GRCh37", 2.0),
        ("COH", "1001", "PGS000002_hmPOS_GRCh37", 3.0), ("COH", "1002", "PGS000002_hmPOS_GRCh37", 4.0),
        ("reference", "HG00096", "PGS000001_hmPOS_GRCh37", 9.0)])
    wide = data.load_aggregated_scores(p, "SUM", sampleset="COH")
    assert list(wide.columns) == ["IID", "PGS000001", "PGS000002"]
    assert wide.set_index("IID").loc["1002", "PGS000002"] == 4.0
    assert "HG00096" not in set(wide["IID"])
    avg = data.load_aggregated_scores(p, "AVG", sampleset="COH")
    assert avg.set_index("IID").loc["1001", "PGS000001"] == pytest.approx(0.1)


def test_avg_derived_when_only_denom(tmp_path):
    df = pd.DataFrame({"sampleset": "COH", "IID": ["1"], "PGS": ["PGS000001_hmPOS_GRCh38"], "SUM": [4.0], "DENOM": [8]})
    df.to_csv(tmp_path / "s.txt.gz", sep="\t", index=False)
    assert data.load_aggregated_scores(tmp_path / "s.txt.gz", "AVG")["PGS000001"].item() == 0.5


def test_legacy_wide_format(tmp_path):
    pd.DataFrame({"sampleset": "COH", "IID": ["1", "2"], "PGS000001_SUM": [1.0, 2.0], "PGS000001_AVG": [0.1, 0.2]}) \
        .to_csv(tmp_path / "w.txt.gz", sep="\t", index=False)
    wide = data.load_aggregated_scores(tmp_path / "w.txt.gz", "SUM")
    assert list(wide.columns) == ["IID", "PGS000001"]


def test_merge_aligns_ids_across_chunks_with_mixed_id_types(tmp_path):
    """Regression: numeric IIDs + reference-panel IIDs must not break alignment."""
    a = write_long(tmp_path / "a.txt.gz", [("COH", "1001", "PGS000001_hmPOS_GRCh37", 1.0),
                                           ("COH", "1002", "PGS000001_hmPOS_GRCh37", 2.0)])
    b = write_long(tmp_path / "b.txt.gz", [("COH", "1001", "PGS000002_hmPOS_GRCh37", 3.0),
                                           ("COH", "1002", "PGS000002_hmPOS_GRCh37", 4.0),
                                           ("reference", "HG00096", "PGS000002_hmPOS_GRCh37", 5.0)])
    merged = data.merge_score_chunks([a, b], "SUM", sampleset="COH")
    assert len(merged) == 2 and not merged.isna().any().any()


def test_merge_rejects_score_in_two_chunks(tmp_path):
    a = write_long(tmp_path / "a.txt.gz", [("COH", "1", "PGS000001_hmPOS_GRCh37", 1.0)])
    b = write_long(tmp_path / "b.txt.gz", [("COH", "1", "PGS000001_hmPOS_GRCh37", 1.0)])
    with pytest.raises(ValueError, match="more than one chunk"):
        data.merge_score_chunks([a, b])


def test_match_summary(tmp_path):
    rows = [("PGS000001_hmPOS_GRCh37", "matched", True, 90), ("PGS000001_hmPOS_GRCh37", "unmatched", True, 10),
            ("PGS000002_hmPOS_GRCh37", "matched", False, 30), ("PGS000002_hmPOS_GRCh37", "excluded", False, 20),
            ("PGS000002_hmPOS_GRCh37", "unmatched", False, 50)]
    pd.DataFrame(rows, columns=["accession", "match_status", "score_pass", "count"]) \
        .assign(sampleset="COH", percent=0).to_csv(tmp_path / "COH_summary.csv", index=False)
    m = data.load_match_summary(tmp_path / "COH_summary.csv").set_index("pgs_id")
    assert m.loc["PGS000001", "match_rate"] == pytest.approx(0.9) and m.loc["PGS000001", "score_pass"]
    assert m.loc["PGS000002", "match_rate"] == pytest.approx(0.3) and not m.loc["PGS000002", "score_pass"]


def test_build_dataset_plink_coding_and_missing(tmp_path):
    feats = pd.DataFrame({"IID": ["1", "2", "3", "4"], "PGS000001": [0.1, 0.2, 0.3, 0.4]})
    (tmp_path / "pheno.txt").write_text("FID IID phenotype\n1 1 2\n2 2 1\n3 3 -9\n4 4 2\n")
    (tmp_path / "cov.txt").write_text("FID IID Sex PC1\n1 1 1 0.1\n2 2 2 0.2\n3 3 1 0.3\n4 4 1 NA\n")
    df, pgs, cov = data.build_dataset(feats, tmp_path / "pheno.txt", tmp_path / "cov.txt")
    assert list(df["IID"]) == ["1", "2"]            # 3: missing phenotype, 4: missing covariate
    assert list(df["phenotype"]) == [1, 0]           # PLINK 2/1 -> 1/0
    assert pgs == ["PGS000001"] and cov == ["Sex", "PC1"]


def test_build_dataset_rejects_non_binary(tmp_path):
    feats = pd.DataFrame({"IID": ["1", "2"], "PGS000001": [0.1, 0.2]})
    (tmp_path / "pheno.txt").write_text("FID IID phenotype\n1 1 0.5\n2 2 3\n")
    (tmp_path / "cov.txt").write_text("FID IID Sex\n1 1 1\n2 2 2\n")
    with pytest.raises(ValueError, match="binary"):
        data.build_dataset(feats, tmp_path / "pheno.txt", tmp_path / "cov.txt")


def _parquet_works() -> bool:
    try:
        pd.DataFrame({"a": [1]}).to_parquet(io.BytesIO())
        return True
    except ImportError:          # missing, or installed but unloadable (e.g. pyarrow>=26 with numpy<2)
        return False


@pytest.mark.skipif(not _parquet_works(), reason="no working parquet engine")
def test_write_frame_parquet_removes_stale_csv(tmp_path):
    stem = tmp_path / "frame"
    stem.with_suffix(".csv").write_text("IID,old\n1,2\n")
    assert data.write_frame(pd.DataFrame({"IID": ["1"], "new": [np.float64(1.0)]}), stem).suffix == ".parquet"
    assert not stem.with_suffix(".csv").exists()
    assert list(data.read_frame(stem).columns) == ["IID", "new"]


def test_write_frame_csv_fallback_removes_stale_parquet(tmp_path, monkeypatch):
    def no_parquet(*_a, **_k):
        raise ImportError("pyarrow requires NumPy 2.0 or newer")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_parquet)
    stem = tmp_path / "frame"
    stem.with_suffix(".parquet").write_text("stale")
    assert data.write_frame(pd.DataFrame({"IID": ["1"], "new": [1.0]}), stem).suffix == ".csv"
    assert not stem.with_suffix(".parquet").exists()
    assert list(data.read_frame(stem).columns) == ["IID", "new"]
