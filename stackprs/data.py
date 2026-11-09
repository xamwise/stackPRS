"""IO helpers: read pgsc_calc output and assemble the modelling dataset.

pgsc_calc (v2.x) facts these helpers are built around (verified against the
v2.3.0 source and its test fixtures):

* scores land in ``<outdir>/<sampleset>/score/aggregated_scores.txt.gz``;
* the file is tab-separated and **long**: ``sampleset, IID, PGS, SUM[, DENOM, AVG]``
  with one row per person per score (``AVG``/``DENOM`` are not always present);
* ``PGS`` holds the harmonised scoring-file name, e.g. ``PGS000018_hmPOS_GRCh37``,
  which must be reduced to the accession to join Catalog metadata;
* with ``--run_ancestry`` the file also contains reference-panel rows;
* per-score matching is summarised in ``<outdir>/<sampleset>/match/<sampleset>_summary.csv``.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd

_ACCESSION = re.compile(r"PGS\d{6}")


def pgs_accession(label) -> str:
    """'PGS000018_hmPOS_GRCh37' -> 'PGS000018' (other names are returned unchanged)."""
    m = _ACCESSION.match(str(label))
    return m.group(0) if m else str(label)


_ID_DTYPES = {"sampleset": str, "FID": str, "IID": str}


def _read_scores_table(path: str | pathlib.Path) -> pd.DataFrame:
    # IDs are always strings: numeric cohort IIDs next to reference-panel IIDs
    # (HG00096, ...) would otherwise parse as int in one chunk and str in another,
    # and the chunk merge would silently fail to align individuals.
    df = pd.read_csv(path, sep="\t", dtype=_ID_DTYPES)
    if df.shape[1] == 1:                  # tolerate a space-delimited variant
        df = pd.read_csv(path, sep=r"\s+", dtype=_ID_DTYPES)
    return df


def load_aggregated_scores(path: str | pathlib.Path, metric: str = "SUM",
                           sampleset: str | None = None) -> pd.DataFrame:
    """Read one ``aggregated_scores.txt.gz`` into a wide ``[IID x PGS]`` frame."""
    metric = metric.upper()
    df = _read_scores_table(path)
    if sampleset is not None and "sampleset" in df.columns:
        df = df[df["sampleset"].astype(str) == str(sampleset)]   # drop reference-panel rows
    if "IID" not in df.columns:
        raise ValueError(f"No IID column in {path}; columns={list(df.columns)[:8]}")

    if "PGS" in df.columns:                                     # long format (pgsc_calc >= 2)
        if metric not in df.columns:
            if metric == "AVG" and {"SUM", "DENOM"} <= set(df.columns):
                df = df.assign(AVG=df["SUM"] / df["DENOM"])
            else:
                raise ValueError(f"{path} has no {metric} column (columns: {list(df.columns)}); "
                                 "use score_value: SUM")
        df = df.assign(PGS=df["PGS"].map(pgs_accession))
        wide = df.pivot(index="IID", columns="PGS", values=metric)   # raises on duplicate IID/PGS
        wide.columns.name = None
        return wide.reset_index()

    # legacy wide format (pgsc_calc 1.x): <PGS>_SUM / <PGS>_AVG columns
    suffix = f"_{metric}"
    cols = [c for c in df.columns if c.endswith(suffix)]
    if not cols:
        raise ValueError(f"Unrecognised pgsc_calc score file {path}; columns={list(df.columns)[:8]}")
    if df["IID"].duplicated().any():
        raise ValueError(f"Duplicate IIDs in {path}; IIDs must be unique")
    return df[["IID"] + cols].rename(columns={c: pgs_accession(c[: -len(suffix)]) for c in cols})


def merge_score_chunks(paths, metric: str = "SUM", sampleset: str | None = None) -> pd.DataFrame:
    """Outer-merge per-chunk score files into one ``[IID x scores]`` matrix."""
    merged: pd.DataFrame | None = None
    for p in paths:
        part = load_aggregated_scores(p, metric=metric, sampleset=sampleset)
        if merged is None:
            merged = part
            continue
        dup = (set(merged.columns) & set(part.columns)) - {"IID"}
        if dup:
            raise ValueError(f"Scores present in more than one chunk (e.g. {sorted(dup)[:3]}); "
                             "stale results? Re-run step 02 and 03.")
        merged = merged.merge(part, on="IID", how="outer")
    if merged is None:
        raise ValueError("No score files supplied to merge.")
    return merged


def load_match_summary(path: str | pathlib.Path) -> pd.DataFrame:
    """Per-score variant matching from ``<sampleset>_summary.csv``.

    Returns one row per accession with ``n_variants``, ``n_matched``,
    ``match_rate`` (matched / all scoring-file variants, as in the pgsc_calc
    report) and ``score_pass`` (passed ``--min_overlap``).
    """
    s = pd.read_csv(path)
    s = s.assign(pgs_id=s["accession"].map(pgs_accession),
                 count=pd.to_numeric(s["count"], errors="coerce").fillna(0),
                 matched=s["match_status"].astype(str).str.lower().eq("matched"),
                 passed=s["score_pass"].astype(str).str.strip().str.lower().isin(["true", "1"]))
    g = s.groupby("pgs_id")
    out = pd.DataFrame({
        "n_variants": g["count"].sum(),
        "n_matched": s[s["matched"]].groupby("pgs_id")["count"].sum(),
        "score_pass": g["passed"].any(),
    }).fillna({"n_matched": 0})
    out["match_rate"] = out["n_matched"] / out["n_variants"].replace(0, np.nan)
    return out.reset_index()


def write_frame(df: pd.DataFrame, stem: pathlib.Path) -> pathlib.Path:
    """Write ``<stem>.parquet`` (``<stem>.csv`` without pyarrow) and delete the
    other variant, so a later :func:`read_frame` can never pick up a stale file."""
    pq, csv = stem.with_suffix(".parquet"), stem.with_suffix(".csv")
    try:
        df.to_parquet(pq, index=False)
        out, stale = pq, csv
    except ImportError as e:
        print(f"  (parquet unavailable: {str(e).splitlines()[0]} — writing {csv.name} instead)")
        df.to_csv(csv, index=False)
        out, stale = csv, pq
    stale.unlink(missing_ok=True)
    return out


def read_frame(stem: pathlib.Path) -> pd.DataFrame:
    """Read what :func:`write_frame` wrote."""
    pq, csv = stem.with_suffix(".parquet"), stem.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {pq.name} nor {csv.name} in {stem.parent} — run the previous step")


def _read_table(path: str | pathlib.Path) -> pd.DataFrame:
    """Read a whitespace- or tab-delimited PLINK-style table."""
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _binary_phenotype(values: pd.Series) -> pd.Series:
    """0/1 case-control coding; accepts PLINK 1/2 (control/case), -9/NA = missing."""
    y = pd.to_numeric(values, errors="coerce").replace(-9, np.nan)
    levels = set(y.dropna().unique())
    if levels <= {0, 1}:
        return y
    if levels <= {1, 2}:
        return y - 1                        # PLINK: 1=control, 2=case
    raise ValueError(f"Phenotype must be binary (0/1 or PLINK 1/2); found levels {sorted(levels)[:6]}")


def build_dataset(feature_matrix: pd.DataFrame,
                  pheno_path: str | pathlib.Path,
                  covariate_path: str | pathlib.Path,
                  pheno_col: str = "phenotype") -> tuple[pd.DataFrame, list[str], list[str]]:
    """Join scores + phenotype + covariates on IID.

    Drops individuals with a missing phenotype or covariate. Returns
    ``(dataframe, pgs_cols, covariate_cols)``; the phenotype column is 0/1.
    """
    pheno = _read_table(pheno_path)
    cov = _read_table(covariate_path)
    if pheno_col not in pheno.columns:
        raise ValueError(f"Column {pheno_col!r} not in {pheno_path} (has {list(pheno.columns)})")

    pgs_cols = [c for c in feature_matrix.columns if c != "IID"]
    feat = feature_matrix[["IID"] + pgs_cols].copy()
    for frame in (feat, pheno, cov):
        frame["IID"] = frame["IID"].astype(str)
    for name, frame in (("scores", feat), ("phenotype", pheno), ("covariate", cov)):
        if frame["IID"].duplicated().any():
            raise ValueError(f"Duplicate IIDs in the {name} table")

    cov_cols = [c for c in cov.columns if c not in ("FID", "IID")]
    pheno = pheno[["IID", pheno_col]].assign(**{pheno_col: _binary_phenotype(pheno[pheno_col])})
    df = (feat.merge(pheno, on="IID", how="inner")
              .merge(cov[["IID"] + cov_cols], on="IID", how="inner"))
    df = df.dropna(subset=[pheno_col] + cov_cols).reset_index(drop=True)
    df[pheno_col] = df[pheno_col].astype(int)
    return df, pgs_cols, cov_cols
