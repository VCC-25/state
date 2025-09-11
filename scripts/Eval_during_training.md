Here is a structured description of your State model extension with Cell-Eval functionality:

## State Model Extension: Cell-Eval Integration

### Overview
You have successfully extended the State model from [ArcInstitute/state](https://github.com/ArcInstitute/state) with a callback for [Cell-Eval](https://github.com/ArcInstitute/cell-eval). This integration enables continuous evaluation during training with specialized metrics for cellular data.

### Configuration Options

#### 1. TOML Configuration File

The extension adds a new `[cell_eval]` section to the TOML configuration file:

```toml
# New: Cell-eval configurations
[cell_eval]
enabled = true                   # Enables/disables Cell-Eval
eval_every_n_steps = 1           # Evaluation every N training steps
plot_every_n_evals = 1           # Plotting every N evaluations
eval_metrics = ['overlap_at_N', 'mae', 'discrimination_score_l1']  # Metrics to use
save_predictions = true          # Saves predictions for further analysis
verbose = true                   # Detailed outputs
```

#### 2. Command Line Parameters (NEW)

In addition to TOML configuration, Cell-Eval parameters can now be set directly via command line:

```bash
+cell_eval.enabled=true \
+cell_eval.eval_every_n_steps=500 \
+cell_eval.plot_every_n_evals=1
```

### Complete Training Example

```bash
uv run state tx train \
  data.kwargs.toml_config_path="examples/training_with_cell_eval.toml" \
  data.kwargs.num_workers=2 \
  data.kwargs.batch_col="batch_var" \
  data.kwargs.pert_col="target_gene" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="non-targeting" \
  data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt" \
  training.max_steps=10000 \
  training.val_freq=10000 \
  training.ckpt_every_n_steps=2000 \
  training.batch_size=2 \
  +model.devices=cuda \
  model=state_sm \
  wandb.tags="[evaluation_test_run_faster]" \
  output_dir="competition" \
  name="evaluation_test_run_faster" \
  +trainer.accelerator=gpu \
  +trainer.devices=1 \
  +trainer.backend=rocm \
  +trainer.precision=16 \
  data.kwargs.pin_memory=true \
  +cell_eval.enabled=true \
  +cell_eval.eval_every_n_steps=500 \
  +cell_eval.plot_every_n_evals=1 \
  2>&1 | tee readable_training.log
```

### Parameter Description

**New Command Line Parameters:**
- **`+cell_eval.enabled=true`**: Activates Cell-Eval during training
- **`+cell_eval.eval_every_n_steps=500`**: Performs evaluation every 500 training steps
- **`+cell_eval.plot_every_n_evals=1`**: Generates plots at every evaluation

**Available Metrics:**
- `overlap_at_N`: Overlap of top-N genes
- `mae`: Mean Absolute Error
- `discrimination_score_l1`: L1-based discrimination score

### Benefits of the Extension

1. **Flexibility**: Configuration possible via both TOML file and command line
2. **Continuous Monitoring**: Evaluation during training without interruption
3. **Specialized Metrics**: Cell-Eval-specific evaluation metrics for cellular data
4. **Visualization**: Automatic plot generation for better interpretability
5. **Logging**: Detailed outputs with `verbose=true` and log saving

This extension makes the State model particularly suitable for evaluating perturbation experiments in cell biology.