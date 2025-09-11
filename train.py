import hydra
from omegaconf import DictConfig

from src.state._cli._tx._train import run_tx_train

# This decorator is equivalent to the hydra.initialize/compose block
# It sets up the config path and loads the main config file.
@hydra.main(config_path="src/state/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Runs the state model training pipeline.

    This function is a Python-based equivalent of the `uv run state tx train`
    command. It uses Hydra to manage configuration. Overrides can be passed
    via the command line.
    """
    run_tx_train(cfg)


if __name__ == "__main__":
    # The command-line arguments from your shell script are passed here.
    # Note that Hydra automatically parses them.
    # Example:
    # python train.py data.kwargs.toml_config_path="examples/fewshot.toml" model=pds_des_optimized
    main()