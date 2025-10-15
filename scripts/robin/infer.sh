
MODEL_DIR="../state-big/models/"

# experiment name
DIR_NAME="few-params-baseline-3"

WANDB_PROJECT="vcc"
WANDB_ENTITY="rsayar728-freie-universit-t-berlin"

# toml config path
TOML_CONFIG="scripts/robin/all_data.toml"

# Competition support set -> why is this important? 
COMPETITION_SUPPORT_SET="../state-big/data"

# perturbation features file
PERT_FEATURES="../state-big/data/ESM2_pert_features.pt"
#"datasets/embeddings/GenePT_gene_embedding_ada_text.pt"
#
COMP_ADATA="../state-big/data/competition_support_set/competition_val_template.h5ad"
CKPT_DIR="${MODEL_DIR}/${DIR_NAME}/checkpoints"
ckpt="${CKPT_DIR}/step=120000.ckpt"
COMP_INFER_DIR="${MODEL_DIR}/${DIR_NAME}/inferred/"
out="${COMP_INFER_DIR}/step=120000.h5ad"
state tx infer \
    --output "$out" \
    --model-dir "${MODEL_DIR}/${DIR_NAME}" \
    --checkpoint "$ckpt" \
    --adata "$COMP_ADATA" \
    --pert-col target_gene

#uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep -i /home/ubuntu/state-big/models/few-params-baseline-3/inferred/step=120000.h5ad -g /home/ubuntu/state-big/data/competition_support_set/gene_names.csv