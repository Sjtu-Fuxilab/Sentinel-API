from pathlib import Path
import os, yaml

ROOT = Path(__file__).resolve().parents[2]
CFG  = yaml.safe_load(open(ROOT / 'configs' / 'default.yaml', 'r', encoding='utf-8'))

def env_or_cfg(key: str, default):
    v = os.getenv(key, None)
    return v if v is not None else default

DATA_ROOT   = Path(env_or_cfg('SENTINEL_DATA',   CFG['paths']['data_root'])).expanduser()
MODELS_DIR  = Path(env_or_cfg('SENTINEL_MODELS', CFG['paths']['models_dir'])).expanduser()
OUTPUTS_DIR = ROOT / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
