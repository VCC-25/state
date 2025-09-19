# Basic checklist (assuming you've ran before with "similar" setup)

- Datasets
    - [x] Embeddings file exists:
        - `PERT_FEATURES="datasets/embeddings/ESM2_pert_features.pt"`
    - [x] `TOML_CONFIG="scripts/robin/all_data.toml"` exists and has your desired training data.

- Output targeting
  - [x] `MODEL_DIR="results/"` and `DIR_NAME=""` are what you intend.
  - [x] Does DIR_NAME exist with previous checkpoints? Load, delete, or move

- Training
    - [x] default steps, ckpts, and val_freq correct?
            training.max_steps=40000 \
            training.ckpt_every_n_steps=500 \
            training.val_freq=250 \
    - [x] learning reate correct? default 1e-4, my preferred is 1e-5
    - [x] checked uv state tx 
        - model set correctly?   model=state_sm \
        - 

- Weights & Biases
  - [x] `use_wandb=false`; if you want logging, set `use_wandb=true` and ensure login/environment.

- Starting ...
  - [] Run from the repo root `state/` (script paths are relative).
  - [ ] Are we starting from "screen" (for compute01)
  - [] sbatch: 



- USING NEW/UNTESTED DATASETs? 
    - [ ] Column names used in the script match your data: `batch_var`, `target_gene`, `cell_type`, `non-targeting`.
    - [ ] Pert features (embeddings) path is correct and matches model expectations:
        - Embedding dims/types align with the model’s `perturbation_features_file` interface.
- NEW/UNTESTED ENVIRONMENT?
     - [ ] threads, num_workers and batch_size correct?
        `THREADS=${THREADS}`, `NUM_WORKERS=${NUM_WORKERS}`, `BATCH_SIZE=${BATCH_SIZE}` 
  


  ------------------------------------------------------------------------------------

  # Continuing from a checkpoint 

- [] remove cp-ing .sh file and change path
- [] remove cp-ing .toml and change path