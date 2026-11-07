"""CivilityLens Configuration Module

Centralized paths, mappings, random seeds, and pipeline parameters
for reproducible cyberbullying detection.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Primary Dataset Paths
RAW_DATA_PATH = DATA_DIR / "cyberbullying_tweets.csv"
FALLBACK_RAW_DATA_PATH = PROJECT_ROOT / "cyberbullying_tweets.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_tweets.parquet"

# Model and Experiment Artifact Paths
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model_pipeline.joblib"
SUBTYPE_MODEL_PATH = ARTIFACTS_DIR / "subtype_model_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
EXPERIMENT_RESULTS_PATH = ARTIFACTS_DIR / "experiment_comparison.csv"
ERROR_ANALYSIS_PATH = ARTIFACTS_DIR / "error_analysis_samples.json"
DATASET_METADATA_PATH = ARTIFACTS_DIR / "dataset_metadata.json"

# Dataset Specification
EXPECTED_COLUMNS = ["tweet_text", "cyberbullying_type"]

ORIGINAL_CLASSES = [
    "religion",
    "age",
    "gender",
    "ethnicity",
    "not_cyberbullying",
    "other_cyberbullying",
]

SUBTYPE_CLASSES = [
    "religion",
    "age",
    "gender",
    "ethnicity",
    "other_cyberbullying",
]

SUBTYPE_DISPLAY_NAMES = {
    "religion": "Religion-based",
    "age": "Age-based",
    "gender": "Gender-based",
    "ethnicity": "Ethnicity/Race-based",
    "other_cyberbullying": "General harassment",
}

# Binary Target Mapping:
# 0 = Non-cyberbullying ('not_cyberbullying')
# 1 = Cyberbullying-related (all other categories)
LABEL_MAPPING = {
    "not_cyberbullying": 0,
    "religion": 1,
    "age": 1,
    "gender": 1,
    "ethnicity": 1,
    "other_cyberbullying": 1,
}

TARGET_NAMES = {
    0: "Non-cyberbullying",
    1: "Cyberbullying-related",
}

# Experimental & Splitting Controls
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Calibrated Moderation Threshold (Accounts for 83% positive class imbalance to reduce false alarms)
DECISION_THRESHOLD = 0.65

# Default Baseline Vectorizer Configuration
DEFAULT_TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 20000,
    "min_df": 2,
    "sublinear_tf": True,
}

DEFAULT_BOW_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 20000,
    "min_df": 2,
}
