#!/usr/bin/env bash
# Step 3 (HPC) — compute the pretrained PRS with pgsc_calc as a Slurm array,
# one task per chunk. Run from the project root AFTER steps 00 and 02:
#
#   python scripts/00_make_samplesheet.py          # writes data/pgsc_params.env
#   python scripts/03_run_pgsc_calc.py --prefetch-only   # on the login node: download scoring
#                                                        # files once (compute nodes may be offline)
#   N=$(ls data/chunks/chunk_*.txt | wc -l)
#   sbatch --array=0-$((N-1)) scripts/03_run_pgsc_calc.slurm
#
# Tune the SBATCH lines, the modules and config/spartan.config for your cluster.
#SBATCH --job-name=stackprs
#SBATCH --array=0-0                 # overridden by --array on the command line
#SBATCH --cpus-per-task=2              # pgsc_calc's heavy steps (format/match) gain little beyond
#SBATCH --mem=12G                      # 2 threads (measured); <= chunk.max_variants keeps memory ~4-8 GB
#SBATCH --time=04:00:00
##SBATCH --partition=cascade        # <-- your partition
#SBATCH --output=data/logs/%x_%a.out
#SBATCH --error=data/logs/%x_%a.err
set -euo pipefail

ROOT="$(pwd)"
mkdir -p "$ROOT/data/logs"
PARAMS="$ROOT/data/pgsc_params.env"
[[ -f "$PARAMS" ]] || { echo "Missing $PARAMS — run scripts/00_make_samplesheet.py"; exit 1; }
# shellcheck source=/dev/null
source "$PARAMS"   # SAMPLESHEET SAMPLESET TARGET_BUILD MIN_OVERLAP PIPELINE_VERSION

# --- cluster modules (edit for your site) -------------------------------------
module load Nextflow/24.04 2>/dev/null || true
module load Singularity 2>/dev/null || true
export NXF_SINGULARITY_CACHEDIR="${NXF_SINGULARITY_CACHEDIR:-$ROOT/data/singularity_cache}"

chunk_id="$(printf 'chunk_%03d' "$SLURM_ARRAY_TASK_ID")"
chunk_file="$ROOT/data/chunks/$chunk_id.txt"
out="$ROOT/data/results/$chunk_id"
[[ -f "$chunk_file" ]] || { echo "No $chunk_file for task $SLURM_ARRAY_TASK_ID"; exit 1; }

# Results are stamped with the ids they were computed for; refuse stale ones.
if [[ -f "$out/pgs_ids.txt" ]] && ! cmp -s "$chunk_file" "$out/pgs_ids.txt"; then
    echo "$out was computed for a different set of scores (filters changed?). Delete it to recompute."
    exit 1
fi
if [[ -f "$out/$SAMPLESET/score/aggregated_scores.txt.gz" ]]; then
    echo "$chunk_id already done, skipping"; exit 0
fi
mkdir -p "$out"
cp "$chunk_file" "$out/pgs_ids.txt"

revision=()
[[ -n "$PIPELINE_VERSION" ]] && revision=(-r "$PIPELINE_VERSION")

# Scoring files: prefetched + staged per chunk if available, else pgsc_calc downloads them.
stage="$ROOT/data/scorefiles/chunks/$chunk_id"
score_args=()
if [[ -d "$stage" ]]; then
    compgen -G "$stage/*.txt.gz" >/dev/null && score_args+=(--scorefile "$stage/*.txt.gz")
    [[ -s "$stage/missing_pgs_ids.txt" ]] && score_args+=(--pgs_id "$(cat "$stage/missing_pgs_ids.txt")")
else
    score_args+=(--pgs_id "$(cat "$chunk_file")")
fi

# pgsc_calc reuses prepared genotypes in <work>/genomes by file name only: drop them
# if the genotype files changed since this chunk's work dir was used (fingerprint from step 00).
work="$ROOT/data/work/$chunk_id"
mkdir -p "$work"
if [[ -d "$work/genomes" ]] && ! cmp -s "$ROOT/data/genotype_fingerprint.txt" "$work/genotype_fingerprint.txt"; then
    echo "Genotype files changed since the last run: removing stale $work/genomes"
    rm -rf "$work/genomes"
fi
cp "$ROOT/data/genotype_fingerprint.txt" "$work/genotype_fingerprint.txt"

# One launch dir per task: concurrent Nextflow runs must not share .nextflow/ state.
launch="$ROOT/data/launch/$chunk_id"
mkdir -p "$launch"
cd "$launch"

echo "Running $chunk_id  (build=$TARGET_BUILD, min_overlap=$MIN_OVERLAP, pgsc_calc=${PIPELINE_VERSION:-latest})"
set +e
nextflow run pgscatalog/pgsc_calc "${revision[@]}" \
    -profile singularity \
    -c "$ROOT/config/spartan.config" \
    -name "stackprs_${chunk_id}_${SLURM_JOB_ID}" \
    -work-dir "$work" \
    --input "$SAMPLESHEET" \
    --target_build "$TARGET_BUILD" \
    "${score_args[@]}" \
    --min_overlap "$MIN_OVERLAP" \
    --max_cpus "${SLURM_CPUS_PER_TASK:-2}" \
    --max_memory "${SLURM_MEM_PER_NODE:-12000}.MB" \
    --outdir "$out" \
    -resume
rc=$?
set -e
if [[ $rc -ne 0 ]] && grep -q "ZeroMatchesError" .nextflow.log 2>/dev/null; then
    echo "No score in $chunk_id reached min_overlap=$MIN_OVERLAP — nothing to compute for this chunk."
    exit 0
fi
exit $rc
