from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SRC_DIR.parent
PROJECT_ROOT = TRAINING_DIR.parent

DATA_RAW_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = TRAINING_DIR / "configs"
OUTPUTS_DIR = TRAINING_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
DEPLOYMENT_MODELS_DIR = PROJECT_ROOT / "deployment" / "flask_app" / "models"


def resolve_config_path(config_arg: str | None) -> Path:
    if config_arg is None:
        raise ValueError("Un fichier de config est requis")

    raw = config_arg.strip()
    direct = Path(raw)
    if direct.exists():
        return direct.resolve()

    fixed = raw.replace("training.configs/", "training/configs/")
    fixed_path = Path(fixed)
    if fixed_path.exists():
        return fixed_path.resolve()

    candidate = PROJECT_ROOT / fixed
    if candidate.exists():
        return candidate.resolve()

    candidate = CONFIGS_DIR / Path(raw).name
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Config introuvable: {config_arg}")
