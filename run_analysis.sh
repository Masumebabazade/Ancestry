#!/usr/bin/env bash
# Run genotype distribution analysis for a given population.
set -euo pipefail

GENO=${1:-head_train_samples.csv}
AIMS=${2:-G_const_infs.csv}
LABELS=${3:-train1.labels}
POP=${4:-ESN}
OUTDIR=${5:-analysis_plots}
THREADS=${6:-$(nproc)}

python analyze_aims.py --genotypes "$GENO" --aims "$AIMS" --labels "$LABELS" \
  --population "$POP" --outdir "$OUTDIR" --threads "$THREADS"

