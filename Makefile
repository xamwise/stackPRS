# stackPRS pipeline. Run `make help` for the stages.
PY := python3
# Override on the command line, e.g. `make run JOBS=8 PROFILE=singularity`.
# When unset, scripts/03_run_pgsc_calc.py uses the `run:` block in config/pipeline.yaml.
JOBS ?=
PROFILE ?=

.PHONY: help env r-deps test samplesheet metadata filter run merge dataset ensemble quick importance analyse all clean

help:
	@echo "stackPRS: pretrained PGS Catalog scores -> disease-specific ensemble"
	@echo "  make env         # create the conda env (environment.yml)"
	@echo "  make r-deps      # install the CRAN uniLasso package (for the unilasso_r learner)"
	@echo "  make test        # offline unit + integration tests"
	@echo "  make metadata    # 01: fetch all PGS Catalog metadata (+ cohort evidence)"
	@echo "  make filter      # 02: cohort exclusions (UKB, All of Us) / ancestry / phenotype filters"
	@echo "  make run         # 03: compute PRS with pgsc_calc, chunks in parallel (JOBS=, PROFILE=)"
	@echo "  make merge       # 04: merge chunk outputs -> feature matrix + match summary"
	@echo "  make dataset     # 05: join phenotype/covariates, score QC, standardise"
	@echo "  make ensemble    # 06: nested evaluation + final uniLasso/AutoGluon ensemble"
	@echo "  make quick       # 06 as a fast smoke run (holdout, AutoGluon 60 s)"
	@echo "  make importance  # 07: coefficients, R-vs-Python uniLasso agreement, EFO enrichment"
	@echo "  make all         # metadata -> ... -> importance"

env:
	conda env create -f environment.yml || conda env update -f environment.yml

r-deps:
	Rscript -e 'install.packages("uniLasso", repos = "https://cloud.r-project.org")'

test:
	$(PY) -m pytest -q tests

samplesheet:
	$(PY) scripts/00_make_samplesheet.py

metadata:
	$(PY) scripts/01_fetch_metadata.py

filter:
	$(PY) scripts/02_filter_and_chunk.py

run: samplesheet
	$(PY) scripts/03_run_pgsc_calc.py $(if $(JOBS),--jobs $(JOBS)) $(if $(PROFILE),--profile $(PROFILE))

merge:
	$(PY) scripts/04_merge_scores.py

dataset:
	$(PY) scripts/05_build_dataset.py

ensemble:
	$(PY) scripts/06_fit_ensemble.py

quick:
	$(PY) scripts/06_fit_ensemble.py --quick

importance:
	$(PY) scripts/07_importance_pleiotropy.py

analyse: merge dataset ensemble importance

all: metadata filter run merge dataset ensemble importance

clean:
	rm -rf data/work data/launch .nextflow*
