#!/bin/bash
# infering for the competition submission and getting the .vcc file

# -- Config --
# Path to your trained model directory
MODEL_DIR="competition"

# experiment name
DIR_NAME="dyno005"

# toml config path
TOML_CONFIG="examples/andrew_few.toml"

# perturbation features file
PERT_FEATURES="competition_support_set/ESM2_pert_features.pt"

# prediction file name
PREDICTION_NAME="prediction"

# checkpoint file name
CKPT="final.ckpt"

# output directory for results
OUT_DIR="cell-eval-outdir"

# parallelization
THREADS=8
NUM_WORKERS=8
BATCH_SIZE=100

# Exit on error
set -e
export OMP_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export HDF5_USE_FILE_LOCKING=FALSE

# Run the training command
uv run state tx train \
  data.kwargs.toml_config_path="${TOML_CONFIG}" \
  data.kwargs.num_workers="${NUM_WORKERS}" \
  data.kwargs.batch_col="batch_var" \
  data.kwargs.pert_col="target_gene" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="non-targeting" \
  data.kwargs.perturbation_features_file="${PERT_FEATURES}" \
  training.batch_size=8 \
  training.max_steps=40000 \
  training.ckpt_every_n_steps=2000 \
  training.val_freq=500 \
  training.lr=1e-4 \
  +training.lr_scheduler="StepLR" \
  +training.lr_step_size=500 \
  +training.lr_gamma=0.95 \
  model=state_sm \
  wandb.tags="[${DIR_NAME}]" \
  output_dir="${MODEL_DIR}" \
  name="${DIR_NAME}" \
  use_wandb=false

# -- Predict --
# gets metrics.csv along with real and predicted adata from test holdouts
uv run state tx infer \
  --output ${MODEL_DIR}/${DIR_NAME}/${PREDICTION_NAME}.h5ad \
  --model_dir ${MODEL_DIR}/${DIR_NAME} \
  --checkpoint ${MODEL_DIR}/${DIR_NAME}/checkpoints/${CKPT} \
  --adata ${COMPETITION_SUPPORT_SET}/competition_val_template.h5ad \
  --pert_col target_gene
  
echo "#### Running cell-eval prep ####"
# remember to have `sudo apt install -y zstd` before running this
uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep -i ${MODEL_DIR}/${DIR_NAME}/${PREDICTION_NAME}.h5ad -g competition_support_set/gene_names.csv
