#!/bin/bash

# -- Activate working environment --
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mne

python scripts/prepare_holdout_ground_truth.py \
  --toml_config examples/andrew_few.toml \
  --split val \
  --output_dir competition/dyno005 \
  --output_h5ad holdout_ground_truth_val.h5ad \
  --output_csv holdout_counts_val.csv