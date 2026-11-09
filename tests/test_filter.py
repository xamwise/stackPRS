import json

import pandas as pd
import pytest

from conftest import load_script

META_COLS = ["pgs_id", "trait_reported", "first_author", "ukb_cohort", "ukb_text_evidence",
             "allofus_cohort", "allofus_text_evidence", "cohorts", "n_variants",
             "gwas_ancestry", "dev_ancestry", "eval_ancestry"]


def write_meta(cfg, rows):
    df = pd.DataFrame(rows, columns=META_COLS)
    df.to_csv(cfg.work_dir / "all_pgs_metadata.csv", index=False)   # CSV round-trip like step 01


FILTERS = {
    "leakage": {"exclude_cohorts": ["UKB", "AllofUs"], "use_text_evidence": True,
                "member_consortia": {"UKB": ["GBMI"]}, "on_unknown_provenance": "drop"},
    "min_variants": 10, "max_variants": 5_000_000,
    "ancestry": {"enabled": True, "target": ["EUR"], "source": "gwas", "min_target_fraction": 80,
                 "on_missing": "drop"},
}

T, F = True, False
ROWS = [
    # pgs_id      trait  auth ukb ukb-text                        aou aou-text cohorts          nvar  gwas
    ("PGS000001", "CRC", "A", T, "", F, "", "UKB|FHS", 100, "EUR:100", "", ""),                       # UKB cohort
    ("PGS000002", "CRC", "A", F, "phenotyping_free: UKB GWAS", F, "", "FHS", 100, "EUR:100", "", ""),  # UKB text
    ("PGS000003", "T2D", "B", F, "", F, "", "GBMI", 100, "EUR:100", "", ""),                          # consortium
    ("PGS000004", "T2D", "B", F, "", F, "", "", 100, "EUR:100", "", ""),                              # no cohorts
    ("PGS000005", "BMI", "C", F, "", F, "", "FHS", 5, "EUR:100", "", ""),                             # few variants
    ("PGS000006", "BMI", "C", F, "", F, "", "BBJ", 100, "EAS:100", "", ""),                           # ancestry
    ("PGS000007", "BMI", "C", F, "", F, "", "FHS", 100, "EUR:60|SAS:40", "", ""),                     # ancestry
    ("PGS000008", "CRC", "D", F, "", F, "", "FHS", 100, "EUR:100", "", ""),                           # KEEP
    ("PGS000009", "LDL", "D", F, "", F, "", "UKBS", 100, "EUR:85|NR:15", "", ""),                     # KEEP (UKBS != UKB)
    ("PGS000010", "BMI", "E", F, "", T, "", "AllofUs|MVP", 100, "EUR:100", "", ""),                   # All of Us
]


def run_filter(make_cfg, filters):
    cfg = make_cfg(filters=filters)
    write_meta(cfg, ROWS)
    load_script("02_filter_and_chunk").main()
    return cfg, pd.read_csv(cfg.work_dir / "filter_decisions.csv").set_index("pgs_id")


def test_filter_funnel_and_reasons(make_cfg):
    cfg, d = run_filter(make_cfg, FILTERS)
    assert sorted(d.index[d.kept]) == ["PGS000008", "PGS000009"]
    assert d.at["PGS000001", "drop_reason"] == "cohort UKB: listed as contributor"
    assert d.at["PGS000002", "drop_reason"] == "cohort UKB: named in sample free text"
    assert d.at["PGS000003", "drop_reason"] == "cohort UKB: via consortium GBMI"
    assert d.at["PGS000004", "drop_reason"] == "no cohort information (provenance unknown)"
    assert d.at["PGS000005", "drop_reason"].startswith("variants <")
    assert d.at["PGS000006", "drop_reason"].startswith("ancestry")
    assert d.at["PGS000007", "drop_reason"].startswith("ancestry")
    assert d.at["PGS000010", "drop_reason"] == "cohort AllofUs: listed as contributor"
    assert (cfg.chunks_dir / "chunk_000.txt").read_text() == "PGS000008,PGS000009"


def test_allofus_exclusion_is_optional(make_cfg):
    filters = json.loads(json.dumps(FILTERS))
    filters["leakage"]["exclude_cohorts"] = ["UKB"]
    _, d = run_filter(make_cfg, filters)
    assert d.at["PGS000010", "kept"]


def test_legacy_config_keys_still_exclude_ukb(make_cfg):
    """A config written before `exclude_cohorts` must not silently keep UKB scores."""
    filters = {**FILTERS, "leakage": {"exclude_ukb": True, "ukb_member_consortia": ["GBMI"],
                                      "on_unknown_provenance": "drop"}}
    _, d = run_filter(make_cfg, filters)
    assert d.at["PGS000001", "drop_reason"] == "cohort UKB: listed as contributor"
    assert d.at["PGS000003", "drop_reason"] == "cohort UKB: via consortium GBMI"
    assert d.at["PGS000010", "kept"]                     # All of Us was not excluded back then


def test_spelling_and_other_cohorts(make_cfg, capsys):
    filters = json.loads(json.dumps(FILTERS))
    filters["leakage"]["exclude_cohorts"] = ["UKB", "All of Us", "MVP", "Finngenn"]
    _, d = run_filter(make_cfg, filters)
    assert d.at["PGS000010", "drop_reason"] == "cohort AllofUs: listed as contributor"   # resolved alias
    out = capsys.readouterr().out
    assert "'MVP' is matched on its Catalog short name only" in out
    assert "WARNING: no score lists a cohort named 'Finngenn'" in out


def test_keep_unknown_provenance(make_cfg):
    filters = json.loads(json.dumps(FILTERS))
    filters["leakage"]["on_unknown_provenance"] = "keep"
    _, d = run_filter(make_cfg, filters)
    assert d.at["PGS000004", "kept"]


# With ancestry disabled, cohort + variant filters leave PGS000006-9.
# Target trait (CRC) = {1, 2, 8} via the ontology; related root adds {9}.
@pytest.mark.parametrize("scope, expected", [
    ("same", {"PGS000008"}),
    ("related", {"PGS000008", "PGS000009"}),
    ("other", {"PGS000006", "PGS000007", "PGS000009"}),
    ("all", {"PGS000006", "PGS000007", "PGS000008", "PGS000009"}),
])
def test_phenotype_scopes_from_cached_ontology(make_cfg, scope, expected):
    cfg = make_cfg(filters={**FILTERS, "ancestry": {"enabled": False},
                            "phenotype": {"scope": scope, "target_traits": ["MONDO_CRC"],
                                          "related_traits": ["MONDO_LIPID"]}})
    write_meta(cfg, ROWS)
    # pre-populated cache == what fetch_trait stores, so no network is needed
    (cfg.cache_dir / "trait_MONDO_CRC.json").write_text(json.dumps(
        {"id": "MONDO_CRC", "associated_pgs_ids": ["PGS000001", "PGS000008"],
         "child_associated_pgs_ids": ["PGS000002"]}))
    (cfg.cache_dir / "trait_MONDO_LIPID.json").write_text(json.dumps(
        {"id": "MONDO_LIPID", "associated_pgs_ids": ["PGS000009"], "child_associated_pgs_ids": []}))
    load_script("02_filter_and_chunk").main()
    clean = pd.read_csv(cfg.work_dir / "pgs_metadata_clean.csv").set_index("pgs_id")
    assert set(clean.index) == expected
    if "PGS000008" in clean.index:
        assert clean.at["PGS000008", "is_target_trait"]


def test_pack_chunks_balances_variant_volume():
    import numpy as np
    pack = load_script("02_filter_and_chunk").pack_chunks
    rng = np.random.default_rng(0)                          # skewed like the Catalog: many tiny, few huge
    sizes = {f"PGS{i:06d}": int(v) for i, v in enumerate(np.concatenate(
        [rng.integers(10, 5_000, 450), rng.integers(100_000, 1_500_000, 50), [4_656_243, 12_000_000]]))}
    chunks = pack(sizes, max_scores=200, max_variants=10_000_000)
    flat = [p for c in chunks for p in c]
    assert sorted(flat) == sorted(sizes) and len(flat) == len(set(flat))      # each score exactly once
    vol = [sum(sizes[p] for p in c) for c in chunks]
    for c, v in zip(chunks, vol):
        assert len(c) <= 200 and (v <= 10_000_000 or len(c) == 1)           # oversized score stands alone
    assert vol == sorted(vol, reverse=True)                                 # heaviest first
    import math                                                             # near-minimal chunk count
    assert len(chunks) <= max(math.ceil(sum(vol) / 10_000_000), math.ceil(len(sizes) / 200)) + 2
    # by count alone (the old behaviour) the volumes would be far more uneven
    by_count = [list(sizes)[i:i + 200] for i in range(0, len(sizes), 200)]
    assert max(vol) <= max(sum(sizes[p] for p in c) for c in by_count)


def test_pack_chunks_without_variant_cap_is_count_based():
    pack = load_script("02_filter_and_chunk").pack_chunks
    chunks = pack({f"PGS{i:06d}": i for i in range(5)}, max_scores=2, max_variants=None)
    assert [len(c) for c in chunks] == [2, 2, 1]
