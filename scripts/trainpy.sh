#! /bin/bash

# Activate your environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mne

# Run the training script
python train.py \
  data.kwargs.toml_config_path="examples/andrewfewshot.toml" \
  data.kwargs.num_workers=4 \
  data.kwargs.batch_col="batch_var" \
  data.kwargs.pert_col="target_gene" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="non-targeting" \
  data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt" \
  training.max_steps=400 \
  training.ckpt_every_n_steps=200 \
  model=state_sm \
  wandb.tags="[first_run]" \
  output_dir="competition" \
  name="first_run" \
  use_wandb=false