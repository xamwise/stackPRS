#!/usr/bin/env python3
"""Step 2 — choose which pretrained PRS to compute, then split them into chunks.

Filters (all configured under `filters:` in pipeline.yaml), applied in order:
  1. cohorts     drop scores whose GWAS / training used an excluded cohort, e.g.
                 UK Biobank (leakage into a UKB target) or All of Us (data use):
                 structured cohort, free-text evidence, member consortia, and
                 optionally scores with unknown provenance
  2. variants    min/max number of variants
  3. ancestry    keep scores derived mostly in your cohort's ancestry
  4. phenotype   all | same (target trait + ontology descendants)
                 | related (target + related roots/categories) | other (not target)

Writes:
  - data/pgs_metadata_clean.csv   kept scores (+ is_target_trait / is_related_trait)
  - data/filter_decisions.csv     every score with the first filter that dropped it
  - data/chunks/chunk_000.txt ... comma-separated PGS ids per pgsc_calc run
"""
import _bootstrap  # noqa: F401

import difflib

import pandas as pd

from stackprs import catalog, config

_ANCESTRY_COL = {"gwas": "gwas_ancestry", "dev": "dev_ancestry", "eval": "eval_ancestry"}
_REQUIRED = ("pgs_id", "trait_reported", "first_author", "cohorts", "n_variants", "gwas_ancestry")


def _truthy(s: pd.Series) -> pd.Series:
    """Bool column robust to 'True'/'False' strings from a CSV round-trip."""
    return s.astype(str).str.strip().str.lower().isin(["true", "1"])


def pack_chunks(sizes: dict[str, int], max_scores: int, max_variants: int | None) -> list[list[str]]:
    """Split scores into pgsc_calc runs of similar *variant volume*.

    pgsc_calc's run time and memory scale with the number of scoring-file
    variants (formatting/matching), not with the number of scores, so packing by
    score count alone leaves one run with most of the work. First-fit decreasing
    on variants (capped by both limits); a score larger than `max_variants` gets a
    chunk of its own. Heaviest chunks come first so they also start first.
    """
    bins: list[list] = []                           # [ids, total variants]
    for pid in sorted(sizes, key=lambda k: (-sizes[k], k)):
        v = sizes[pid]
        for b in bins:
            if len(b[0]) < max_scores and (max_variants is None or b[1] + v <= max_variants):
                b[0].append(pid)
                b[1] += v
                break
        else:
            bins.append([[pid], v])
    bins.sort(key=lambda b: (-b[1], b[0][0]))
    return [sorted(b[0]) for b in bins]


def excluded_cohorts(leak: dict) -> list[str]:
    """Configured cohort exclusions. Older configs used `exclude_ukb` (default
    true); it is still honoured so such a config never silently keeps UKB."""
    names = leak.get("exclude_cohorts")
    if names is None:
        names = ["UKB"] if leak.get("exclude_ukb", True) else []
    names = [str(n) for n in names]
    if leak.get("exclude_ukb") and not any(catalog.known_cohort(n) is catalog.KNOWN_COHORTS["UKB"] for n in names):
        names.insert(0, "UKB")
    return names


def member_consortia(leak: dict, name: str) -> set[str]:
    """Consortia configured as including `name` (+ the older `ukb_member_consortia`)."""
    spec = catalog.known_cohort(name)
    def same(key: str) -> bool:
        return catalog.known_cohort(key) is spec if spec else key.lower() == name.lower()
    out = {c.upper() for key, cs in (leak.get("member_consortia") or {}).items() if same(key) for c in cs or []}
    if spec is catalog.KNOWN_COHORTS["UKB"]:
        out |= {c.upper() for c in leak.get("ukb_member_consortia") or []}
    return out


def main() -> None:
    cfg = config.load()
    f = cfg.filters
    df = pd.read_csv(cfg.work_dir / "all_pgs_metadata.csv")
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Metadata lacks {missing} — re-run scripts/01_fetch_metadata.py")

    reason = pd.Series("", index=df.index)       # first filter that dropped each score
    funnel = [("PGS Catalog scores", len(df))]

    def drop(mask: pd.Series, label: str) -> None:
        reason[mask & reason.eq("")] = label
        funnel.append((label, int(reason.eq("").sum())))

    # 1. excluded cohorts (leakage, data use) ---------------------------------
    leak = f.get("leakage") or {}
    excluded = excluded_cohorts(leak)
    cohort_sets = df["cohorts"].fillna("").map(lambda s: {c.upper() for c in s.split("|") if c})
    for name in excluded:
        spec = catalog.known_cohort(name)
        label = spec.key if spec else name
        if spec:
            p = spec.key.lower()
            if f"{p}_cohort" not in df.columns:
                raise SystemExit(f"Metadata has no {spec.key} evidence columns — re-run "
                                 "scripts/01_fetch_metadata.py (add --offline to reuse the cached records)")
            drop(_truthy(df[f"{p}_cohort"]), f"cohort {label}: listed as contributor")
            if leak.get("use_text_evidence", True):
                drop(df[f"{p}_text_evidence"].fillna("").str.len() > 0, f"cohort {label}: named in sample free text")
        else:
            listed = cohort_sets.map(lambda s, n=name.upper(): n in s)
            if not listed.any():
                names = sorted({c for s in df["cohorts"].fillna("") for c in s.split("|") if c} | set(catalog.KNOWN_COHORTS))
                close = difflib.get_close_matches(name, names, n=3, cutoff=0.6)
                print(f"WARNING: no score lists a cohort named {name!r}"
                      + (f" — did you mean {close}?" if close else " (use Catalog short names)"))
            else:
                print(f"Note: {name!r} is matched on its Catalog short name only "
                      f"(free-text evidence exists for {list(catalog.KNOWN_COHORTS)})")
            drop(listed, f"cohort {label}: listed as contributor")
        consortia = member_consortia(leak, name)
        if consortia:
            drop(cohort_sets.map(lambda s, c=consortia: bool(s & c)),
                 f"cohort {label}: via consortium {'/'.join(sorted(consortia))}")
    if excluded:
        policy = leak.get("on_unknown_provenance", "drop")
        if policy not in ("drop", "keep"):
            raise SystemExit("filters.leakage.on_unknown_provenance must be 'drop' or 'keep'")
        if policy == "drop":
            drop(df["cohorts"].fillna("").eq(""), "no cohort information (provenance unknown)")

    # 2. variants -------------------------------------------------------------
    nvar = pd.to_numeric(df["n_variants"], errors="coerce").fillna(0)
    if "min_variants" in f:
        drop(nvar < f["min_variants"], f"variants < {f['min_variants']}")
    if "max_variants" in f:
        drop(nvar > f["max_variants"], f"variants > {f['max_variants']}")

    # 3. ancestry -------------------------------------------------------------
    anc = f.get("ancestry") or {}
    if anc.get("enabled"):
        source = anc.get("source", "gwas")
        col = _ANCESTRY_COL.get(source)
        if col is None:
            raise SystemExit(f"filters.ancestry.source must be one of {list(_ANCESTRY_COL)}; got {source!r}")
        targets = anc.get("target", "EUR")
        min_frac = float(anc.get("min_target_fraction", 0))
        keep_missing = anc.get("on_missing", "drop") == "keep"

        def passes(cell) -> bool:
            dist = catalog.parse_ancestry(cell)
            if not dist:                       # ancestry group not reported for this score
                return keep_missing
            return catalog.ancestry_target_fraction(dist, targets) >= min_frac

        drop(~df[col].apply(passes), f"ancestry: {targets} < {min_frac:g}% of {source} samples")

    # 4. phenotype ------------------------------------------------------------
    ph = f.get("phenotype") or {}
    scope = ph.get("scope", "all")
    if scope not in ("all", "same", "related", "other"):
        raise SystemExit("filters.phenotype.scope must be one of all | same | related | other")
    target_traits = ph.get("target_traits") or []
    related_traits = ph.get("related_traits") or []
    related_categories = ph.get("related_categories") or []
    if scope != "all" and not target_traits:
        raise SystemExit(f"filters.phenotype.scope={scope!r} needs filters.phenotype.target_traits")
    target_set = catalog.trait_pgs_ids(target_traits, cfg.api_base, cfg.cache_dir)
    related_roots = list(related_traits) + catalog.category_trait_ids(related_categories, cfg.api_base, cfg.cache_dir)
    related_set = target_set | catalog.trait_pgs_ids(related_roots, cfg.api_base, cfg.cache_dir)
    df["is_target_trait"] = df["pgs_id"].isin(target_set)
    df["is_related_trait"] = df["pgs_id"].isin(related_set)
    if scope == "same":
        drop(~df["is_target_trait"], "phenotype: not the target trait")
    elif scope == "related":
        drop(~df["is_related_trait"], "phenotype: not target or related")
    elif scope == "other":
        drop(df["is_target_trait"], "phenotype: target trait (scope=other)")

    # outputs -----------------------------------------------------------------
    kept = reason.eq("")
    evidence = [c for c in df.columns if c.endswith("_text_evidence")]
    decisions = df[["pgs_id", "trait_reported", "first_author", "cohorts", *evidence,
                    "is_target_trait", "is_related_trait"]].assign(kept=kept, drop_reason=reason)
    decisions.to_csv(cfg.work_dir / "filter_decisions.csv", index=False)
    clean = df[kept].copy()
    clean.to_csv(cfg.work_dir / "pgs_metadata_clean.csv", index=False)

    ids = clean["pgs_id"].tolist()
    size, max_variants = cfg.chunk_size, cfg.get("chunk.max_variants")
    sizes = dict(zip(clean["pgs_id"], pd.to_numeric(clean["n_variants"], errors="coerce").fillna(0).astype(int)))
    for old in cfg.chunks_dir.glob("chunk_*.txt"):
        old.unlink()
    chunks = pack_chunks(sizes, size, int(max_variants) if max_variants else None)
    for i, chunk in enumerate(chunks):
        (cfg.chunks_dir / f"chunk_{i:03d}.txt").write_text(",".join(chunk))

    n0 = funnel[0][1]
    print("Filter funnel (scores remaining after each filter):")
    for label, n in funnel:
        print(f"  {label:<48} {n:>6}  ({n / n0:6.1%})")
    if target_traits:
        tgt = df["is_target_trait"]
        by_cohort = (tgt & (reason.str.startswith("cohort ") | reason.str.startswith("no cohort"))).sum()
        print(f"\nTarget-trait scores ({', '.join(target_traits)}): {int(tgt.sum())} in Catalog, "
              f"{int(by_cohort)} removed by cohort exclusions ({', '.join(excluded) or 'none'}), "
              f"{int((tgt & kept).sum())} kept")
    volumes = [sum(sizes[p] for p in c) / 1e6 for c in chunks]
    print(f"\nKept {len(ids)} scores ({sum(volumes):.1f} M scoring variants) -> {len(chunks)} chunks "
          f"(<= {size} scores" + (f", <= {int(max_variants) / 1e6:g} M variants" if max_variants else "")
          + f"; largest {max(volumes, default=0):.1f} M, smallest {min(volumes, default=0):.1f} M)")
    print(f"  per-score decisions: {cfg.work_dir / 'filter_decisions.csv'}")


if __name__ == "__main__":
    main()
