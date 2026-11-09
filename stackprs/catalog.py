"""PGS Catalog REST client: score metadata, cohort evidence, trait resolution.

Cohort evidence (leakage and data-use exclusions)
-------------------------------------------------
The Catalog records, per score, the samples used to derive the variant weights
(``samples_variants`` = the source GWAS) and to develop/tune the score
(``samples_training``). A score is excluded when a listed cohort contributed to
either — UK Biobank because it is the target cohort (leakage), All of Us e.g.
for data-use reasons. For the cohorts in :data:`KNOWN_COHORTS` we collect three
kinds of evidence (audited against the full Catalog):

1. **Structured cohorts** — exact ``cohorts[].name_short`` (``UKB``,
   ``AllofUs``) or the full name. Matching is exact on purpose: ``UKBS`` (UK
   National Blood Donor Service) and ``UKBGS`` (UK Breakthrough Generations
   Study) are *not* UK Biobank, so a ``"UKB" in name`` test would wrongly
   exclude them.
2. **Free text** in the sample records (``phenotyping_free``, ``source_DOI``,
   ``cohorts_additional``, ...). ~200 scores trained on UKB GWAS (Neale lab,
   SAIGE UKBB, "UKB GWAS (ICD10-based phenotype): C18" ...) only say so here —
   including colorectal-cancer PRS. Negated mentions ("samples from UK Biobank
   excluded", "UKB removed", "without UKB") mark the *clean* versions of such
   scores and are ignored. (All of Us currently appears only as a structured
   cohort; the text pattern guards future records.)
3. **Consortia** whose meta-analyses include a cohort (e.g. GBMI includes UKB) —
   checked in the filter step from the stored cohort names (configurable).

Other cohorts can be excluded by their Catalog short name (structured evidence
only). Scores that list no cohorts at all have unknown provenance; the filter
step decides whether to keep them.
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_API = "https://www.pgscatalog.org/rest"
SAMPLE_GROUPS = ("samples_variants", "samples_training")


def make_session() -> requests.Session:
    """requests session that retries transient API failures with backoff."""
    retry = Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    sess = requests.Session()
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    return sess


def iter_scores(api_base: str = DEFAULT_API, page_size: int = 250,
                session: requests.Session | None = None) -> Iterator[dict[str, Any]]:
    """Yield every score object in the Catalog, following pagination."""
    sess = session or make_session()
    url: str | None = f"{api_base.rstrip('/')}/score/all"
    params: dict[str, Any] | None = {"limit": page_size}
    while url:
        r = sess.get(url, params=params, timeout=120)
        r.raise_for_status()
        payload = r.json()
        yield from payload.get("results", [])
        url = payload.get("next")   # already carries limit/offset
        params = None


# --------------------------------------------------------------------------- #
# Cohort evidence
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CohortSpec:
    """How a cohort appears in Catalog records."""
    key: str                    # config name; metadata columns are <key.lower()>_cohort / _text_evidence
    short_names: frozenset      # exact cohorts[].name_short values (upper-cased)
    full_name: re.Pattern       # searched in cohorts[].name_full
    text: re.Pattern            # free-text evidence in the sample records


KNOWN_COHORTS: dict[str, CohortSpec] = {
    # "UK Biobank", "UKB", "UKBB", "UKB-EUR", "UKBB_SAIGE", Neale lab UKB GWAS URLs.
    # Not UKBS / UKBGS (word boundary after UKB/UKBB).
    "UKB": CohortSpec(
        key="UKB", short_names=frozenset({"UKB"}),
        full_name=re.compile(r"uk biobank", re.I),
        text=re.compile(r"uk\s*-?\s*biobank|\bukbb?(?:\b|[-_])|nealelab|pan-?ukb", re.I)),
    # Catalog record: name_short "AllofUs", name_full "All of Us Research Program | NIH".
    # "AoU" only in that capitalisation: all-caps "AOU" is also the Italian
    # "Azienda Ospedaliero-Universitaria" (hospital) in cohort descriptions.
    "AllofUs": CohortSpec(
        key="AllofUs", short_names=frozenset({"ALLOFUS"}),
        full_name=re.compile(r"\ball\s*of\s*us\b", re.I),
        text=re.compile(r"(?i:\ball\s*-?\s*of\s*-?\s*us\b)|\bAoU\b")),
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_ALIASES = {"ukbiobank": "UKB", "ukbb": "UKB", "aou": "AllofUs", "allofusresearchprogram": "AllofUs"}


def known_cohort(name: str) -> CohortSpec | None:
    """Registry entry for a configured cohort name, tolerant of spelling:
    "UKB", "UK Biobank", "AllofUs", "All of Us" and "AoU" all resolve."""
    n = _norm(name)
    key = next((k for k in KNOWN_COHORTS if _norm(k) == n), None) or _ALIASES.get(n)
    return KNOWN_COHORTS.get(key) if key else None


def _samples(score: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for group in SAMPLE_GROUPS:
        yield from score.get(group) or []


def cohort_matches(cohort: dict[str, Any], spec: CohortSpec) -> bool:
    """Exact short-name or full-name match — e.g. UKB but not UKBS/UKBGS."""
    short = (cohort.get("name_short") or "").strip().upper()
    return short in spec.short_names or bool(spec.full_name.search(cohort.get("name_full") or ""))


def score_cohorts(score: dict[str, Any]) -> list[str]:
    """Short names of all cohorts that contributed to the GWAS or score training."""
    return [c["name_short"] for s in _samples(score)
            for c in s.get("cohorts") or [] if c.get("name_short")]


def cohort_listed(score: dict[str, Any], spec: CohortSpec) -> bool:
    """Structured evidence: the cohort is listed as a contributor."""
    return any(cohort_matches(c, spec) for s in _samples(score) for c in s.get("cohorts") or [])


# Negations must sit right next to the mention ("without UKB", "UKB removed",
# "UK Biobank samples were excluded") and not across a clause boundary, so that
# "UKB, related samples removed" still counts as use of the cohort.
_NEG_BEFORE = re.compile(r"\b(?:without|excluding|except|minus|not\s+including|removing|non)\b[^,;.]{0,20}$", re.I)
_NEG_AFTER = re.compile(r"^[^,;.]{0,30}?\b(?:excluded|removed|omitted|dropped)\b", re.I)


def _is_negated(text: str, match: re.Match) -> bool:
    before = text[max(0, match.start() - 40): match.start()]
    after = text[match.end(): match.end() + 60]
    return bool(_NEG_BEFORE.search(before) or _NEG_AFTER.search(after))


def text_evidence(score: dict[str, Any], spec: CohortSpec) -> str:
    """First non-negated free-text mention of the cohort in the sample records.

    Returns ``"<field>: <text>"`` or ``""``. Only the GWAS / training sample
    records are scanned — not e.g. ``method_params``, where UKB typically appears
    as an LD reference panel (genotypes only, no phenotype leakage).
    """
    for sample in _samples(score):
        for field, value in sample.items():
            if field == "cohorts" or not isinstance(value, str):
                continue
            if any(not _is_negated(value, m) for m in spec.text.finditer(value)):
                return f"{field}: {value[:160]}"
    return ""


# --------------------------------------------------------------------------- #
# Flattening for the metadata table
# --------------------------------------------------------------------------- #
def _dist_str(dist: dict[str, Any]) -> str:
    """'EUR:80|NR:20' encoding of an ancestry percentage distribution."""
    return "|".join(f"{k}:{v}" for k, v in dist.items())


def flatten_score(score: dict[str, Any]) -> dict[str, Any]:
    """One metadata row per score; '|' joins multi-valued fields."""
    efos = score.get("trait_efo") or []
    pub = score.get("publication") or {}
    anc = score.get("ancestry_distribution") or {}

    def dist(group: str) -> str:
        return _dist_str(((anc.get(group) or {}).get("dist")) or {})

    row = {
        "pgs_id": score.get("id"),
        "name": score.get("name"),
        "trait_reported": score.get("trait_reported"),
        "trait_efo_ids": "|".join(e.get("id", "") for e in efos),
        "trait_efo_labels": "|".join(e.get("label", "") for e in efos),
        "n_variants": score.get("variants_number"),
        "variants_genomebuild": score.get("variants_genomebuild"),
        "weight_type": score.get("weight_type"),
        "method_name": score.get("method_name"),
        "pgp_id": pub.get("id"),
        "first_author": pub.get("firstauthor"),
        "publication_date": pub.get("date_publication"),
        "cohorts": "|".join(sorted(set(score_cohorts(score)))),
    }
    for spec in KNOWN_COHORTS.values():          # ukb_cohort, ukb_text_evidence, allofus_cohort, ...
        prefix = spec.key.lower()
        row[f"{prefix}_cohort"] = cohort_listed(score, spec)
        row[f"{prefix}_text_evidence"] = text_evidence(score, spec)
    row.update({
        # Ancestry percentage distributions per sample group (GWAS Catalog codes).
        "gwas_ancestry": dist("gwas"),  # source GWAS — most relevant for transfer
        "dev_ancestry": dist("dev"),    # score-development samples
        "eval_ancestry": dist("eval"),  # evaluation samples
        "date_release": score.get("date_release"),
    })
    return row


# --------------------------------------------------------------------------- #
# Ancestry parsing helpers
# --------------------------------------------------------------------------- #
def parse_ancestry(value: Any) -> dict[str, float]:
    """Parse an 'EUR:80|NR:20' string back into {code: percent}.

    Tolerates None/NaN/empty (returns {}) so it can be applied to a column
    straight from the metadata CSV.
    """
    out: dict[str, float] = {}
    if not value or not isinstance(value, str):
        return out
    for part in value.split("|"):
        code, sep, val = part.rpartition(":")
        if not sep:
            continue
        try:
            out[code] = float(val)
        except ValueError:
            continue
    return out


def ancestry_target_fraction(dist: dict[str, float], targets) -> float:
    """Total percentage of `targets` (a code or list of codes) in `dist`."""
    if isinstance(targets, str):
        targets = [targets]
    return float(sum(dist.get(t, 0.0) for t in targets))


# --------------------------------------------------------------------------- #
# Trait (phenotype) resolution via the Catalog's ontology
# --------------------------------------------------------------------------- #
def _cached_get(url: str, cache_file: pathlib.Path, session: requests.Session | None,
                params: dict[str, Any] | None = None) -> Any:
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    sess = session or make_session()
    r = sess.get(url, params=params, timeout=120)
    if r.status_code == 404:
        raise ValueError(f"PGS Catalog returned 404 for {url}")
    r.raise_for_status()
    payload = r.json()
    if payload:   # the API answers unknown ids with HTTP 200 + {} — never cache that
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload))
    return payload


def fetch_trait(trait_id: str, api_base: str, cache_dir: pathlib.Path,
                session: requests.Session | None = None) -> dict[str, Any]:
    """Trait record incl. PGS of all ontology descendants (``include_children=1``)."""
    t = _cached_get(f"{api_base.rstrip('/')}/trait/{trait_id}",
                    cache_dir / f"trait_{trait_id}.json", session, {"include_children": 1})
    if not isinstance(t, dict) or "id" not in t:
        raise ValueError(f"Unknown trait id {trait_id!r} in the PGS Catalog "
                         "(use EFO/MONDO/HP ids such as MONDO_0005575)")
    return t


def trait_pgs_ids(trait_ids: Iterable[str], api_base: str, cache_dir: pathlib.Path,
                  session: requests.Session | None = None) -> set[str]:
    """All PGS annotated to the traits or any of their ontology descendants.

    This is the same expansion pgsc_calc's ``--efo_id`` uses.
    """
    ids: set[str] = set()
    for tid in trait_ids:
        t = fetch_trait(tid, api_base, cache_dir, session)
        ids |= set(t.get("associated_pgs_ids") or []) | set(t.get("child_associated_pgs_ids") or [])
    return ids


def category_trait_ids(categories: Iterable[str], api_base: str, cache_dir: pathlib.Path,
                       session: requests.Session | None = None) -> list[str]:
    """Trait ids in the named PGS Catalog trait categories (case-insensitive).

    Note: the Catalog has no "Cancer" category (colorectal cancer sits under
    "Digestive system disorder"/"Other disease"); use ontology roots such as
    MONDO_0004992 (cancer) for cancer-wide sets instead.
    """
    categories = list(categories)
    if not categories:
        return []
    payload = _cached_get(f"{api_base.rstrip('/')}/trait_category/all",
                          cache_dir / "trait_categories.json", session)
    cats = payload.get("results", payload) if isinstance(payload, dict) else payload
    by_label = {c["label"].lower(): c for c in cats}
    wanted = [c.lower() for c in categories]
    unknown = [c for c in wanted if c not in by_label]
    if unknown:
        raise ValueError(f"Unknown trait categories {unknown}; available: {sorted(c['label'] for c in cats)}")
    return [t["id"] for c in wanted for t in by_label[c].get("efotraits") or []]
