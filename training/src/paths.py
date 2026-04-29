"""Shared project paths for the training scripts.

This keeps folder locations in one place instead of repeating hardcoded paths
in every file.
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SRC_DIR.parent
PROJECT_ROOT = TRAINING_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = TRAINING_DIR / "configs"
OUTPUTS_DIR = TRAINING_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
DEPLOYMENT_MODELS_DIR = PROJECT_ROOT / "deployment" / "flask_app" / "models"


def resolve_config_path(config_arg: str) -> Path:
    text = config_arg.strip().replace("training.configs/", "training/configs/")

    possible_paths = [
        Path(text),
        PROJECT_ROOT / text,
        TRAINING_DIR / text,
        CONFIGS_DIR / Path(text).name,
    ]

    for path in possible_paths:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"Config introuvable: {config_arg}")
