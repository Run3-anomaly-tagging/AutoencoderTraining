import os
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "dataset_config.json")
DEFAULT_MERGED_QCD_FILE = os.path.join(DATA_DIR, "merged_qcd_train.h5")

DEFAULT_SIGNAL_FILE = ""
with open(DEFAULT_CONFIG_PATH, 'r') as f:
    config = json.load(f)
    DEFAULT_SIGNAL_FILE = config["signal_samples"]["GluGluHto2B"]["path"]

DENSE_AUTOENCODER_MODEL = os.path.join(MODELS_DIR, "dense_autoencoder", "best_model.h5")
IMAGE_AUTOENCODER_MODEL = os.path.join(MODELS_DIR, "image_autoencoder", "best_model.h5")
