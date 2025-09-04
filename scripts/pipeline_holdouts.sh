#!/bin/bash
# infering for the competition submission and getting the .vcc file

# -- Config --
# Path to your trained model directory
MODEL_DIR="competition_experiments"

# experiment name
DIR_NAME="training_finetune_andrew3"

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

# Exit on error
set -e
export OMP_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export HDF5_USE_FILE_LOCKING=FALSE

# -- Activate working environment --
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mne

echo "#### Running training ####"

uv run state tx train \
  data.kwargs.toml_config_path=${TOML_CONFIG} \
  data.kwargs.num_workers=${NUM_WORKERS} \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file=${PERT_FEATURES} \
  training.max_steps=2000 \
  training.ckpt_every_n_steps=500 \
  training.val_freq=250 \
  model=state_sm \
  model.kwargs.nb_decoder=true \
  wandb.tags=[${DIR_NAME}] \
  output_dir=${MODEL_DIR} \
  name=${DIR_NAME} \
  use_wandb=false

echo "### Generating holdout ground truth ###"
python scripts/prepare_holdout_ground_truth.py \
  --toml_config ${TOML_CONFIG} \
  --split test \
  --output_dir ${OUT_DIR} \
  --output_h5ad holdout_ground_truth_test.h5ad \
  --output_csv holdout_counts_test.csv

echo "#### Running cell-eval baseline ####"

uv run -m cell_eval baseline \
    -a ${OUT_DIR}/${DIR_NAME}/holdout_ground_truth_test.h5ad \
    -c ${OUT_DIR}/${DIR_NAME}/holdout_counts_test.csv \
    -o ${OUT_DIR}/${DIR_NAME}/baseline_test.h5ad \
    -O ${OUT_DIR}/${DIR_NAME}/baseline_de_test.csv \
    --pert-col target_gene \
    --control-pert non-targeting \
    --num-threads ${THREADS} 

echo "#### Running prediction ####"

# gets metrics.csv along with real and predicted adata from test holdouts
uv run state tx predict \
    --checkpoint "final.ckpt" \
    --output_dir ${MODEL_DIR}/${DIR_NAME} \
    --profile full

echo "#### Running inference ####"

# gets prediction.h5ad for the holdout predictions
uv run state tx infer \
  --output ${MODEL_DIR}/${DIR_NAME}/${PREDICTION_NAME}.h5ad \
  --model_dir ${MODEL_DIR}/${DIR_NAME} \
  --checkpoint ${MODEL_DIR}/${DIR_NAME}/checkpoints/${CKPT} \
  --adata ${OUT_DIR}/${DIR_NAME}/holdout_ground_truth_test.h5ad \
  --pert_col target_gene

echo "#### Running cell-eval run ####"

# run cell-eval on the holdout predictions
uv run -m cell_eval run \
    -ap ${MODEL_DIR}/${DIR_NAME}/${PREDICTION_NAME}.h5ad \
    -ar ${OUT_DIR}/${DIR_NAME}/holdout_ground_truth_test.h5ad \
    -o ${OUT_DIR}/${DIR_NAME}/cell-eval-outdir-results \
    --pert-col target_gene \
    --control-pert non-targeting \
    --profile vcc \
    --skip-metrics pearson_edistance,clustering_agreement,discrimination_score_l2,discrimination_score_cosine \
    --num-threads ${THREADS} \
    --batch-size ${BATCH_SIZE}

# run cell-eval on the baseline predictions
uv run -m cell_eval run \
    -ap ${OUT_DIR}/${DIR_NAME}/baseline_test.h5ad \
    -ar ${OUT_DIR}/${DIR_NAME}/holdout_ground_truth_test.h5ad \
    -o ${OUT_DIR}/${DIR_NAME}/cell-eval-outdir-baseline \
    --pert-col target_gene \
    --control-pert non-targeting \
    --profile vcc \
    --skip-metrics pearson_edistance,clustering_agreement,discrimination_score_l2,discrimination_score_cosine \
    --num-threads ${THREADS} \
    --batch-size ${BATCH_SIZE}

echo "#### Running cell-eval score ####"

uv run -m cell_eval score \
    -i ${OUT_DIR}/${DIR_NAME}/cell-eval-outdir-results/agg_results.csv \
    -I ${OUT_DIR}/${DIR_NAME}/cell-eval-outdir-baseline/agg_results.csv \
    -o ${OUT_DIR}/${DIR_NAME}/baseline_diff_test.csv

# gets prediction.h5ad for the holdout predictions
uv run state tx infer \
  --output ${MODEL_DIR}/${DIR_NAME}/competition_val_prediction.h5ad \
  --model_dir ${MODEL_DIR}/${DIR_NAME} \
  --checkpoint ${MODEL_DIR}/${DIR_NAME}/checkpoints/${CKPT} \
  --adata competition_support_set/competition_val_template.h5ad \
  --pert_col target_gene

echo "#### Running cell-eval prep ####"
# remember to have `sudo apt install -y zstd` before running this
uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep -i ${MODEL_DIR}/${DIR_NAME}/competition_val_prediction.h5ad -g competition_support_set/gene_names.csv