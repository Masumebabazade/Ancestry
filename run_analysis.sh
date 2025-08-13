#!/usr/bin/env bash
# Run genotype distribution analysis.
# By default all populations are processed; pass a population code as the
# sixth argument to limit the analysis.  Provide any value as a seventh
# argument to also generate plots.
set -euo pipefail

GENO=${1:-head_train_samples.csv}
AIMS=${2:-G_const_infs.csv}
LABELS=${3:-train1.labels}
OUTDIR=${4:-analysis_plots}
THREADS=${5:-$(nproc)}
POP=${6:-}
PLOTS=${7:-}

if [[ -n "$PLOTS" ]]; then
  PLOTS_FLAG="--plots"
else
  PLOTS_FLAG=""
fi

if [[ -n "$POP" ]]; then
  python analyze_aims.py --genotypes "$GENO" --aims "$AIMS" --labels "$LABELS" \
    --population "$POP" --outdir "$OUTDIR" --threads "$THREADS" $PLOTS_FLAG
else
  python analyze_aims.py --genotypes "$GENO" --aims "$AIMS" --labels "$LABELS" \
    --outdir "$OUTDIR" --threads "$THREADS" $PLOTS_FLAG
fi

