import json

import pytest

from stackprs import catalog as C

UKB, AOU = C.KNOWN_COHORTS["UKB"], C.KNOWN_COHORTS["AllofUs"]


@pytest.mark.parametrize("cohort, spec, expected", [
    ({"name_short": "UKB", "name_full": "UK Biobank"}, UKB, True),
    ({"name_short": "X", "name_full": "The UK Biobank study"}, UKB, True),
    ({"name_short": "UKBS", "name_full": "UK National Blood Donor Service"}, UKB, False),
    ({"name_short": "UKBGS", "name_full": "UK Breakthrough Generations Study"}, UKB, False),
    ({"name_short": "BioMe", "name_full": "BioMe BioBank"}, UKB, False),
    # the Catalog's actual All of Us record
    ({"name_short": "AllofUs", "name_full": "All of Us Research Program | National Institutes of Health (NIH)"}, AOU, True),
    ({"name_short": "X", "name_full": "NIH All of Us cohort"}, AOU, True),
    ({"name_short": "UKB", "name_full": "UK Biobank"}, AOU, False),
    ({"name_short": "AOU-CAREGGI", "name_full": "Azienda Ospedaliero-Universitaria Careggi"}, AOU, False),
])
def test_cohort_matching_is_exact(cohort, spec, expected):
    assert C.cohort_matches(cohort, spec) is expected


@pytest.mark.parametrize("name, key", [
    ("UKB", "UKB"), ("ukb", "UKB"), ("UK Biobank", "UKB"), ("UKBB", "UKB"),
    ("AllofUs", "AllofUs"), ("All of Us", "AllofUs"), ("AoU", "AllofUs"),
    ("All of Us Research Program", "AllofUs"), ("FinnGen", None),
])
def test_known_cohort_tolerates_spelling(name, key):
    spec = C.known_cohort(name)
    assert (spec.key if spec else None) == key


# Real phrasings from the PGS Catalog (audited over all 6,991 scores).
@pytest.mark.parametrize("text, leaky", [
    ("UKB GWAS (ICD10-based phenotype): C18", True),
    ("http://www.nealelab.is/uk-biobank", True),
    ("ftp://share.sph.umich.edu/UKBB_SAIGE_HRC/", True),
    ("from all UK Biobank participants who did not undergo cMRI", True),
    ("UKB, related samples removed", True),
    ("UK Biobank (excluding related individuals)", True),
    ("Original GWAS study (GCST005839) with samples from UK Biobank excluded", False),
    ("UKB removed from the original analysis", False),
    ("without UKB", False),
    ("meta-analysis excluding UK Biobank", False),
    ("BSTOP, Estonian Biobank, HUNT, UK Biobank removed", False),
    ("non-UKB cohorts", False),
    ("UKBS", False),
    ("UKBGS", False),
])
def test_ukb_text_evidence(text, leaky):
    score = {"samples_variants": [{"cohorts": [], "phenotyping_free": text}]}
    assert bool(C.text_evidence(score, UKB)) is leaky


# The Catalog has no free-text All of Us mentions today; these guard future records.
@pytest.mark.parametrize("text, used", [
    ("GWAS in the All of Us Research Program (v7)", True),
    ("summary statistics from AllofUs", True),
    ("AoU participants of European ancestry", True),
    ("meta-analysis of MVP and All-of-Us", True),
    ("excluding All of Us", False),
    ("All of Us participants were removed", False),
    ("recruited at AOU Careggi, Florence", False),        # Italian university hospital, not All of Us
    ("UK Biobank GWAS", False),
])
def test_allofus_text_evidence(text, used):
    score = {"samples_training": [{"cohorts": [], "cohorts_additional": text}]}
    assert bool(C.text_evidence(score, AOU)) is used


def test_text_scan_ignores_cohort_field_and_method_params():
    score = {"method_params": "LD reference: UK Biobank",
             "samples_variants": [{"cohorts": [{"name_short": "UKBS", "name_full": "UK Biobank-like"}]}]}
    assert C.text_evidence(score, UKB) == ""


def test_flatten_score():
    s = {"id": "PGS000001", "trait_efo": [{"id": "MONDO_1", "label": "cancer"}], "variants_number": 77,
         "publication": {"id": "PGP1", "firstauthor": "A", "date_publication": "2019-01-01"},
         "samples_training": [{"cohorts": [{"name_short": "UKB", "name_full": "UK Biobank"}]}],
         "samples_variants": [{"cohorts": [{"name_short": "ABCFS"}], "source_DOI": "nealelab.is"}],
         "ancestry_distribution": {"gwas": {"dist": {"EUR": 80, "NR": 20}}}}
    row = C.flatten_score(s)
    assert row["ukb_cohort"] is True
    assert row["ukb_text_evidence"].startswith("source_DOI")
    assert row["allofus_cohort"] is False and row["allofus_text_evidence"] == ""
    assert row["cohorts"] == "ABCFS|UKB"
    assert row["gwas_ancestry"] == "EUR:80|NR:20" and row["dev_ancestry"] == ""
    assert row["publication_date"] == "2019-01-01"


def test_parse_ancestry():
    assert C.parse_ancestry("EUR:80|NR:20") == {"EUR": 80.0, "NR": 20.0}
    assert C.parse_ancestry("") == {} and C.parse_ancestry(None) == {} and C.parse_ancestry(float("nan")) == {}
    assert C.ancestry_target_fraction({"EUR": 60, "SAS": 40}, "EUR") == 60
    assert C.ancestry_target_fraction({"EUR": 60, "SAS": 40}, ["EUR", "SAS"]) == 100


def test_trait_resolution_from_cache(tmp_path):
    (tmp_path / "trait_MONDO_1.json").write_text(json.dumps(
        {"id": "MONDO_1", "associated_pgs_ids": ["PGS000001"], "child_associated_pgs_ids": ["PGS000002"]}))
    assert C.trait_pgs_ids(["MONDO_1"], "https://example.invalid", tmp_path) == {"PGS000001", "PGS000002"}
    (tmp_path / "trait_categories.json").write_text(json.dumps(
        {"results": [{"label": "Digestive system disorder", "efotraits": [{"id": "MONDO_1"}]}]}))
    assert C.category_trait_ids(["digestive system disorder"], "https://example.invalid", tmp_path) == ["MONDO_1"]
    with pytest.raises(ValueError, match="Unknown trait categories"):
        C.category_trait_ids(["Cancer"], "https://example.invalid", tmp_path)
    assert C.category_trait_ids([], "https://example.invalid", tmp_path) == []


def test_unknown_trait_is_an_error_and_not_cached(tmp_path, monkeypatch):
    class Resp:                       # the live API answers unknown ids with 200 + {}
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {}
    class Sess:
        def get(self, *a, **k): return Resp()
    with pytest.raises(ValueError, match="Unknown trait id"):
        C.fetch_trait("MONDO_9999999", "https://example.invalid", tmp_path, session=Sess())
    assert not (tmp_path / "trait_MONDO_9999999.json").exists()
