"""stackPRS — a disease-specific ensemble on top of pretrained PRS.

Pretrained polygenic scores from the PGS Catalog (filtered for leakage,
ancestry and phenotype) are computed per individual with pgsc_calc; an ensemble
of uniLasso (Python + R), AutoGluon and lasso is then stacked on top of them to
predict one disease, with interpretation via coefficients and trait enrichment.
"""

__version__ = "0.2.0"
