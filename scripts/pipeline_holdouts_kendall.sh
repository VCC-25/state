#!/bin/bash
# infering for the competition submission and getting the .vcc file

# -- Config --
# Path to your trained model directory
MODEL_DIR="competition"

# experiment name
DIR_NAME="kendall_de"

# toml config path
TOML_CONFIG="examples/fewshot.toml"

# prep TOML
PREP_TOML_CONFIG="examples/fewshot_prep.toml"

# Competition support set
COMPETITION_SUPPORT_SET="/raid/kreid/v_cell/competition_support_set"

# perturbation features file
PERT_FEATURES="/raid/kreid/v_cell/competition_support_set/ESM2_pert_features.pt"

# prediction file name
PREDICTION_NAME="prediction"

# output directory for results
OUT_DIR=cell-eval-outdir/${DIR_NAME}

# parallelization
THREADS=8
NUM_WORKERS=8
BATCH_SIZE=64

# Exit on error
set -e
export OMP_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export HDF5_USE_FILE_LOCKING=FALSE

echo "#### Running training ####"

uv run state tx train \
  data.kwargs.toml_config_path=${TOML_CONFIG} \
  data.kwargs.num_workers=${NUM_WORKERS} \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file=${PERT_FEATURES} \
  training.max_steps=1000 \
  training.ckpt_every_n_steps=2500 \
  training.val_freq=50 \
  training.lr=1e-5 \
  model=state_sm \
  model.kwargs.nb_decoder=false \
  +model.kwargs.differential_expression_loss=true \
  +model.kwargs.ranking_loss=false \
  wandb.tags=[${DIR_NAME}] \
  output_dir=${MODEL_DIR} \
  name=${DIR_NAME} \
  use_wandb=false

uv run scripts/prepare_holdout_ground_truth.py \
  --toml_config ${PREP_TOML_CONFIG} \
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

# get just the checkpoint filename
CKPT=$(basename $(ls ${MODEL_DIR}/${DIR_NAME}/checkpoints/*val_loss*.ckpt | head -n 1))
echo "Using checkpoint filename: $CKPT"


echo "#### Running prediction ####"

# gets metrics.csv along with real and predicted adata from test holdouts
uv run state tx predict \
    --checkpoint "${CKPT}" \
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
    --batch-size ${BATCH_SIZE} \
    --profile vcc

# run cell-eval on the baseline predictions
uv run -m cell_eval run \
    -ap ${OUT_DIR}/baseline_test.h5ad \
    -ar ${OUT_DIR}/holdout_ground_truth_test.h5ad \
    -o ${OUT_DIR}/cell-eval-outdir-baseline \
    --pert-col target_gene \
    --control-pert non-targeting \
    --skip-metrics pearson_edistance,clustering_agreement,discrimination_score_l2,discrimination_score_cosine \
    --num-threads ${THREADS} \
    --batch-size ${BATCH_SIZE} \
    --profile vcc

echo "#### Running cell-eval score ####"

uv run -m cell_eval score \
    -i ${OUT_DIR}/cell-eval-outdir-results/agg_results.csv \
    -I ${OUT_DIR}/cell-eval-outdir-baseline/agg_results.csv \
    -o ${OUT_DIR}/baseline_diff_test.csv

# # gets prediction.h5ad for the holdout predictions
# # uv run state tx infer \
# #   --output ${MODEL_DIR}/${DIR_NAME}/competition_val_prediction.h5ad \
# #   --model_dir ${MODEL_DIR}/${DIR_NAME} \
# #   --checkpoint ${MODEL_DIR}/${DIR_NAME}/checkpoints/${CKPT} \
# #   --adata ${COMPETITION_SUPPORT_SET}/competition_val_template.h5ad \
# #   --pert_col target_gene

# echo "#### Running cell-eval prep ####"
# # # remember to have `sudo apt install -y zstd` before running this
# # uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep -i ${MODEL_DIR}/${DIR_NAME}/competition_val_prediction.h5ad -g ${COMPETITION_SUPPORT_SET}/gene_names.csv
