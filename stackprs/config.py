"""Load the central pipeline configuration (config/pipeline.yaml).

Paths in the YAML are interpreted relative to the project root (the directory
that contains the ``stackprs`` package), so the config stays portable as long
as the data sits where ``pipeline.yaml`` says it does.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"


def _resolve(p: str | pathlib.Path) -> pathlib.Path:
    """Resolve a possibly-relative path against the project root."""
    p = pathlib.Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _mkdir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Nested lookup, e.g. ``cfg.get("ensemble.autogluon.time_limit", 300)``."""
        node: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node or node[key] is None:
                return default
            node = node[key]
        return node

    # --- paths ---
    @property
    def plink_prefix(self) -> pathlib.Path:
        return _resolve(self.raw["paths"]["plink_prefix"])

    @property
    def pheno(self) -> pathlib.Path:
        return _resolve(self.raw["paths"]["pheno"])

    @property
    def covariate(self) -> pathlib.Path:
        return _resolve(self.raw["paths"]["covariate"])

    @property
    def work_dir(self) -> pathlib.Path:
        return _mkdir(_resolve(self.raw["paths"]["work_dir"]))

    @property
    def results_dir(self) -> pathlib.Path:
        return _mkdir(self.work_dir / "results")

    @property
    def chunks_dir(self) -> pathlib.Path:
        return _mkdir(self.work_dir / "chunks")

    @property
    def cache_dir(self) -> pathlib.Path:
        return _mkdir(self.work_dir / "cache")

    @property
    def ensemble_dir(self) -> pathlib.Path:
        return _mkdir(self.work_dir / "ensemble")

    # --- target genotypes ---
    @property
    def build(self) -> str:
        return self.raw["target"]["build"]

    @property
    def sampleset(self) -> str:
        return self.raw["target"]["sampleset"]

    @property
    def plink_format(self) -> str:
        return self.get("target.format", "bfile")

    @property
    def pheno_col(self) -> str:
        return self.get("target.pheno_col", "phenotype")

    # --- catalog / filters / chunking ---
    @property
    def api_base(self) -> str:
        return self.raw["catalog"]["api_base"].rstrip("/")

    @property
    def page_size(self) -> int:
        return int(self.get("catalog.page_size", 250))

    @property
    def filters(self) -> dict[str, Any]:
        return self.raw.get("filters", {}) or {}

    @property
    def chunk_size(self) -> int:
        return int(self.raw["chunk"]["size"])

    @property
    def score_value(self) -> str:
        """Which pgsc_calc column to use as the per-person score: SUM or AVG."""
        return str(self.get("score_value", "SUM")).upper()

    # --- pgsc_calc runs (local parallel runner + Slurm) ---
    @property
    def run_profile(self) -> str:
        return self.get("run.profile", "docker")

    @property
    def extra_config(self) -> pathlib.Path | None:
        """Optional extra Nextflow -c config (e.g. an HPC profile)."""
        v = self.get("run.extra_config")
        return _resolve(v) if v else None

    @property
    def min_overlap(self) -> float:
        """pgsc_calc --min_overlap: scores matching fewer variants are dropped."""
        return float(self.get("run.min_overlap", 0.75))

    @property
    def pipeline_version(self) -> str | None:
        """pgsc_calc release passed as ``-r`` (None = whatever is pulled)."""
        return self.get("run.pipeline_version")


def load(path: str | pathlib.Path | None = None) -> Config:
    path = pathlib.Path(path) if path else DEFAULT_CONFIG
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)
