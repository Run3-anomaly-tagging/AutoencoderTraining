import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "dataset_config.json")
DEFAULT_MERGED_QCD_FILE = os.path.join(DATA_DIR, "merged_qcd_train.h5")