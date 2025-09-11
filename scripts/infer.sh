#!/bin/bash

# Run the inference command
# # NOTE: model_dir looks for a ./checkpoints/final.ckpt to infer on
# uv run state tx infer \
#   --model_dir "competition/dyno005" \
#   --adata competition/dyno005/holdout_ground_truth_val.h5ad \
#   --output competition/dyno005/prediction_val.h5ad \
#   --pert_col "target_gene"

# # Then evaluate
# uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval \
#     -ap competition/dyno005/prediction_val.h5ad \
#     -ar competition/dyno005/holdout_ground_truth_val.h5ad \
#     -o competition/dyno005/results \
#     --pert-col target_gene \
#     --control-pert non-targeting \
#     --num-threads 8 \
#     --batch-size 16 \
#     --profile vcc

uv run -m cell_eval run \
    -ap competition/dyno005/prediction_val.h5ad \
    -ar competition/dyno005/holdout_ground_truth_val.h5ad \
    -o competition/dyno005/results \
    --pert-col target_gene \
    --control-pert non-targeting \
    --profile vcc \
    --skip-metrics pearson_edistance,clustering_agreement,discrimination_score_l2,discrimination_score_cosine \
    --num-threads 8 \
    --batch-size 16