#!/usr/bin/env Rscript
# Fit cv.uniLasso (CRAN package 'uniLasso') for a binary outcome and write the
# coefficients at the chosen lambda. Called by stackprs.learners.UniLassoR.
#
# Data exchange is raw little-endian float64 (X column-major), so nothing beyond
# 'uniLasso' (and its dependency 'glmnet') is required.
#
# usage: Rscript unilasso_fit.R <workdir> <n> <p> <nfolds> <seed> <lambda.min|lambda.1se>
# in:    <workdir>/X.bin (n*p doubles), <workdir>/y.bin (n doubles, 0/1)
# out:   <workdir>/coef.bin (p+1 doubles: intercept, then slopes), <workdir>/fit.json

if (!requireNamespace("uniLasso", quietly = TRUE))
  stop("R package 'uniLasso' is not installed: install.packages('uniLasso')")
suppressPackageStartupMessages(library(uniLasso))

a      <- commandArgs(trailingOnly = TRUE)
wd     <- a[1]
n      <- as.integer(a[2])
p      <- as.integer(a[3])
nfolds <- as.integer(a[4])
seed   <- as.integer(a[5])
s      <- a[6]
stopifnot(s %in% c("lambda.min", "lambda.1se"))

x <- matrix(readBin(file.path(wd, "X.bin"), "double", n = n * p, endian = "little"), nrow = n, ncol = p)
y <- readBin(file.path(wd, "y.bin"), "double", n = n, endian = "little")
stopifnot(length(y) == n, all(y %in% c(0, 1)), !anyNA(x))

set.seed(seed)
fit <- cv.uniLasso(x, y, family = "binomial", nfolds = nfolds)

cf <- as.numeric(as.matrix(coef(fit, s = s)))   # final (gamma) coefficients on the original x scale
writeBin(cf, file.path(wd, "coef.bin"), endian = "little")
lam <- if (s == "lambda.1se") fit$lambda.1se else fit$lambda.min
writeLines(sprintf('{"lambda": %.12g, "n_nonzero": %d, "uniLasso_version": "%s"}',
                   lam, sum(cf[-1] != 0), as.character(packageVersion("uniLasso"))),
           file.path(wd, "fit.json"))
