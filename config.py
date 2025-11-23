"""
Configuration and Constants for Music Genre Classification Pipeline

This module centralizes all project configurations, paths, and MLflow setup
that are shared across different workloads (data, training, evaluation).
"""

import os
import random
from pathlib import Path

import numpy as np
import mlflow


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "fma_small"
METADATA_DIR = PROJECT_ROOT / "fma_metadata"
FEATURES_DIR = PROJECT_ROOT / "preprocessed_features"
MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
FEATURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA SETTINGS
# ============================================================================

MAX_TRACKS = 8000
SAMPLE_RATE = 22050         # Audio sample rate (Hz)
AUDIO_DURATION = 30         # Use first 30 seconds of each track

# Feature extraction settings
N_MFCC = 20                 # Number of MFCC coefficients
N_CHROMA = 12               # Number of chroma features
N_SPECTRAL_CONTRAST = 6     # Spectral contrast bands (6 to avoid Nyquist error)

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Minimum samples per genre
MIN_SAMPLES_PER_GENRE = 50

# Random seed for reproducibility
RANDOM_SEED = 42


# ============================================================================
# MODEL SETTINGS
# ============================================================================

LGBM_PARAMS = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': -1
}

N_ESTIMATORS = 200
EARLY_STOPPING_ROUNDS = 20


# ============================================================================
# RAY SETTINGS
# ============================================================================

RAY_NUM_CPUS = 7            # Number of CPUs for Ray (adjust based on your system)
RAY_NUM_TRIALS = 5          # Number of hyperparameter tuning trials
RAY_MAX_CONCURRENT = 2      # Max concurrent trials


# ============================================================================
# MLFLOW SETTINGS
# ============================================================================

MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT_NAME = "music-genre-ray"

# Setup MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across numpy and Python's random module.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_audio_path(track_id: int, data_dir: Path = DATA_DIR) -> Path:
    """
    Get the file path for an audio track given its ID.
    
    FMA dataset organizes tracks in subdirectories based on the first 3 digits of track ID.
    Example: track 123456 → fma_medium/123/123456.mp3
    
    Args:
        track_id: The track ID
        data_dir: Root directory of audio files
        
    Returns:
        Path object pointing to the audio file
    """
    track_id_str = f"{track_id:06d}"
    subdir = track_id_str[:3]
    filename = f"{track_id_str}.mp3"
    return data_dir / subdir / filename


# Set seed on module import
set_seed()

print(f"Configuration loaded")
print(f"  Project root: {PROJECT_ROOT}")
print(f"  MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"  Random seed: {RANDOM_SEED}")
