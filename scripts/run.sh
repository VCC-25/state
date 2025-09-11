#!/bin/bash

MODEL_DIR="competition"
DIR_NAME="dyno005"
PREDICTION_NAME="prediction"
THREADS=8
BATCH_SIZE=16

uv run -m cell_eval run \
    -ap ${MODEL_DIR}/${DIR_NAME}/${PREDICTION_NAME}.h5ad \
    -ar ${MODEL_DIR}/${DIR_NAME}/holdout_ground_truth_val.h5ad \
    -o ${MODEL_DIR}/${DIR_NAME}/results \
    --pert-col target_gene \
    --control-pert non-targeting \
    --num-threads ${THREADS} \
    --profile vcc \
    --batch-size ${BATCH_SIZE}