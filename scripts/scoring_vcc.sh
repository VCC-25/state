#!/bin/bash
# infering for the competition submission and getting the .vcc file

# -- Config --
# Path to your trained model directory
MODEL_DIR="competition"

# experiment name
DIR_NAME="training_finetune_andrew1"

# toml config path
TOML_CONFIG="examples/andrew_fewshot.toml"

# perturbation features file
PERT_FEATURES="competition_support_set/ESM2_pert_features.pt"

# prediction file name
PREDICTION_NAME="prediction"

# checkpoint file name
CKPT="final.ckpt"

# output directory for results
OUT_DIR=cell-eval-outdir

# parallelization
THREADS=8
NUM_WORKERS=8
BATCH_SIZE=100

# -- Activate working environment --
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mne

# Exit on error
set -e
export OMP_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export HDF5_USE_FILE_LOCKING=FALSE



uv run -m cell_eval score \
    -i ${OUT_DIR}/cell-eval-outdir-results/agg_results.csv \
    -I ${OUT_DIR}/cell-eval-outdir-baseline/agg_results.csv \
    -o ${OUT_DIR}/baseline_diff_test.csv

echo "#### Running cell-eval prep ####"
# remember to have `sudo apt install -y zstd` before running this
uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep -i ${MODEL_DIR}/${DIR_NAME}/${PREDICTION_NAME}.h5ad -g competition_support_set/gene_names.csv
