#!/bin/bash
# infering for the competition submission and getting the .vcc file

# -- Config --
# Path to your trained model directory where your models are
MODEL_DIR="results/"

# experiment name
DIR_NAME="esm2-cell-set-len-256-3-en-and-decoder-layers"

# toml config path
TOML_CONFIG="scripts/robin/all_data.toml"

# Competition support set -> why is this important? 
COMPETITION_SUPPORT_SET="datasets/base_dataset"

# perturbation features file
PERT_FEATURES="datasets/embeddings/ESM2_pert_features.pt"
#"datasets/embeddings/GenePT_gene_embedding_ada_text.pt"
#
# prediction file name
PREDICTION_NAME="prediction"

# checkpoint file name
CKPT="final.ckpt"

# output directory for results
OUT_DIR=results/pipeline_holdouts_robin

# parallelization
THREADS=8
NUM_WORKERS=8
BATCH_SIZE=100

# Exit on error
set -e
export OMP_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export HDF5_USE_FILE_LOCKING=FALSE


# Copy this script into the model directory for reproducibility
DEST_DIR="${MODEL_DIR}/${DIR_NAME}"
mkdir -p "$DEST_DIR"

# Path of the currently running script (works when executed or sourced)
SRC="${BASH_SOURCE[0]:-$0}"

# Resolve to an absolute path if possible
if command -v realpath >/dev/null 2>&1; then
  SRC_ABS=$(realpath "$SRC")
elif command -v readlink >/dev/null 2>&1; then
  SRC_ABS=$(readlink -f "$SRC" 2>/dev/null || printf "%s" "$SRC")
else
  SRC_ABS="$SRC"
fi

cp -a "$SRC_ABS" "$DEST_DIR/" || cp -a "$SRC" "$DEST_DIR/"


# copy toml config into the model directory for reproducibility
if [ -e "$TOML_CONFIG" ]; then
  cp -a "$TOML_CONFIG" "$DEST_DIR/$(basename "$TOML_CONFIG")"
else
  echo "Warning: TOML_CONFIG '$TOML_CONFIG' does not exist. Skipping copy." >&2
fi

echo "#### Running training ####"

#use learning rate 1e-5
uv run state tx train \
  data.kwargs.toml_config_path=${TOML_CONFIG} \
  data.kwargs.num_workers=${NUM_WORKERS} \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file=${PERT_FEATURES} \
  training.max_steps=40000 \
  training.ckpt_every_n_steps=400 \
  training.val_freq=200 \
  model=state_sm \
  model.kwargs.nb_decoder=true \
  wandb.tags=[${DIR_NAME}] \
  output_dir=${MODEL_DIR} \
  name=${DIR_NAME} \
  training.lr=1e-5 \
  use_wandb=false \
  model.kwargs.n_encoder_layers=3 \
  model.kwargs.n_decoder_layers=3 \
  model.kwargs.cell_set_len=256 \


#model.kwargs.transformer_backbone_kwargs.num_hidden_layers=2 \


python scripts/prepare_holdout_ground_truth.py \
  --toml_config ${TOML_CONFIG} \
  --split test \
  --output_dir ${OUT_DIR} \
  --output_h5ad holdout_ground_truth_test.h5ad \
  --output_csv holdout_counts_test.csv

echo "#### Running cell-eval baseline ####"

uv run -m cell_eval baseline \
    -a ${OUT_DIR}/holdout_ground_truth_test.h5ad \
    -c ${OUT_DIR}/holdout_counts_test.csv \
    -o ${OUT_DIR}/baseline_test.h5ad \
    -O ${OUT_DIR}/baseline_de_test.csv \
    --pert-col target_gene \
    --control-pert non-targeting \
    --num-threads ${THREADS} 

echo "#### Running prediction ####"

# gets metrics.csv along with real and predicted adata from test holdouts
uv run state tx predict \
    --checkpoint "final.ckpt" \
    --output_dir "${MODEL_DIR}/${DIR_NAME}/" \
    --profile full

echo "#### Running inference ####"

# gets prediction.h5ad for the holdout predictions
uv run state tx infer \
  --output ${MODEL_DIR}/${DIR_NAME}/${PREDICTION_NAME}.h5ad \
  --model_dir ${MODEL_DIR}/${DIR_NAME} \
  --checkpoint ${MODEL_DIR}/${DIR_NAME}/checkpoints/${CKPT} \
  --adata ${OUT_DIR}/holdout_ground_truth_test.h5ad \
  --pert_col target_gene

echo "#### Running cell-eval run ####"

# run cell-eval on the holdout predictions
uv run -m cell_eval run \
    -ap ${MODEL_DIR}/${DIR_NAME}/${PREDICTION_NAME}.h5ad \
    -ar ${OUT_DIR}/holdout_ground_truth_test.h5ad \
    -o ${OUT_DIR}/cell-eval-outdir-results \
    --pert-col target_gene \
    --control-pert non-targeting \
    --skip-metrics pearson_edistance,clustering_agreement,discrimination_score_l2,discrimination_score_cosine \
    --num-threads ${THREADS} \
    --batch-size ${BATCH_SIZE}

# run cell-eval on the baseline predictions
uv run -m cell_eval run \
    -ap ${OUT_DIR}/baseline_test.h5ad \
    -ar ${OUT_DIR}/holdout_ground_truth_test.h5ad \
    -o ${OUT_DIR}/cell-eval-outdir-baseline \
    --pert-col target_gene \
    --control-pert non-targeting \
    --skip-metrics pearson_edistance,clustering_agreement,discrimination_score_l2,discrimination_score_cosine \
    --num-threads ${THREADS} \
    --batch-size ${BATCH_SIZE}

echo "#### Running cell-eval score ####"

uv run -m cell_eval score \
    -i ${OUT_DIR}/cell-eval-outdir-results/agg_results.csv \
    -I ${OUT_DIR}/cell-eval-outdir-baseline/agg_results.csv \
    -o ${OUT_DIR}/baseline_diff_test.csv

# gets prediction.h5ad for the holdout predictions
uv run state tx infer \
  --output ${MODEL_DIR}/${DIR_NAME}/competition_val_prediction.h5ad \
  --model_dir ${MODEL_DIR}/${DIR_NAME} \
  --checkpoint ${MODEL_DIR}/${DIR_NAME}/checkpoints/${CKPT} \
  --adata ${COMPETITION_SUPPORT_SET}/competition_val_template.h5ad \
  --pert_col target_gene

echo "#### Running cell-eval prep ####"
# # remember to have `sudo apt install -y zstd` before running this
# uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep -i ${MODEL_DIR}/${DIR_NAME}/competition_val_prediction.h5ad -g ${COMPETITION_SUPPORT_SET}/gene_names.csv
