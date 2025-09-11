MODEL_DIR="results/competition/hvgs_2_state_sm_esm2"


# prediction file name
PREDICTION_NAME="prediction"

# checkpoint file name
#last.ckpt if interrupted, final.ckpt if finished
CKPT="last.ckpt"

# output directory for results
OUT_DIR=cell-eval-outdir

# gets prediction.h5ad for the holdout predictions
uv run state tx infer \
  --output ${MODEL_DIR}/${PREDICTION_NAME}.h5ad \
  --model_dir ${MODEL_DIR} \
  --checkpoint ${MODEL_DIR}/checkpoints/${CKPT} \
  --adata datasets/competition_support_set/hvgs/preprocessed_competition_val_template.h5ad \
  --pert_col target_gene

#i don't get it, if we provide the adata, why do we even need to infer?

