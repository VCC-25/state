set -euo pipefail
#-> preprocess_infer
#-> infer
#-> cell-eval run 

INPUT_DATASET="datasets/base_dataset/hvgs/preprocessed_training_data_jurkat.h5ad"
DATASET_NAME="preprocessed_training_data_jurkat"
MODEL_DIR="results/hvgs_2_state_sm_esm2"

PROCESSED_DATASET=$MODEL_DIR/prep_infer_preprocessed_jurkat.h5ad

CHECKPOINT_NAME="last.ckpt"





#testing preproccess_infer prepped_competition_train.h5ad -> infer
#uv run state tx preprocess_infer \
#  --adata $INPUT_DATASET \
#  --output $PROCESSED_DATASET \
#  --pert_col target_gene \
#  --control_condition non-targeting

OUT_DIR="$MODEL_DIR/predictions"
mkdir -p $OUT_DIR
PREDICTION="$OUT_DIR/$DATASET_NAME.h5ad"

#------------------------------------------------------------------------------------------
#Output the prediction file name in plain text based on the above variables: 
#e.g. PREDICTION -> "$OUT_DIR/$DATASET_NAME.h5ad" 
#   -> results/hvgs_2_state_sm_esm2/predictions/preprocessed_training_data_jurkat.h5ad
#PROCESSED_DATASET 
#   -> results/hvgs_2_state_sm_esm2/prep_infer_preprocessed_jurkat.h5ad
#OUT_DIR 
#   -> results/hvgs_2_state_sm_esm2/predictions
#------------------------------------------------------------------------------------------

#INFER
#uv run state tx infer \
#  --output $PREDICTION \
#  --model_dir $MODEL_DIR \
#  --checkpoint "$MODEL_DIR/checkpoints/$CHECKPOINT_NAME" \
#  --adata $PROCESSED_DATASET \
#  --pert_col target_gene \
#  --embed_key X_hvg
#output: results/hvgs_2_state_sm_esm2/predictions/preprocessed_training_data_jurkat.h5ad

# run cell-eval on the holdout predictions
#cell-eval run \
#    -ap $PREDICTION \
#    -ar $INPUT_DATASET \
#    -o ${OUT_DIR}/cell-eval-$DATASET_NAME/results \
#    --pert-col target_gene \
#    --control-pert non-targeting \
#    --skip-metrics pearson_edistance,clustering_agreement,discrimination_score_l2,discrimination_score_cosine \
#    --num-threads 12 
# --batch-size ${BATCH_SIZE}
#output: results/hvgs_2_state_sm_esm2/predictions/cell-eval-preprocessed_training_data_jurkat/results


# run cell-eval on the baseline predictions
#processed_dataset as ap because it is random
cell-eval run \
    -ap $PROCESSED_DATASET \
    -ar $INPUT_DATASET \
    -o ${OUT_DIR}/cell-eval-$DATASET_NAME/baseline \
    --pert-col target_gene \
    --control-pert non-targeting \
    --skip-metrics pearson_edistance,clustering_agreement,discrimination_score_l2,discrimination_score_cosine \
    --num-threads 12
#--batch-size ${BATCH_SIZE}
#output: results/hvgs_2_state_sm_esm2/predictions/cell-eval-preprocessed_training_data_jurkat/baseline

echo "#### Running cell-eval score ####"

cell-eval score \
    -i ${OUT_DIR}/cell-eval-$DATASET_NAME/results/agg_results.csv \
    -I ${OUT_DIR}/cell-eval-$DATASET_NAME/baseline/agg_results.csv \
    -o ${OUT_DIR}/cell-eval-scoring-$DATASET_NAME.csv