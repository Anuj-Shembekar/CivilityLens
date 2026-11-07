"""Dataset Loader and Validator for CivilityLens.

Handles Stage 1 (Loading), Stage 2 (Cleaning & Validation), and Stage 3 (Target Preparation).
Maintains original categories while mapping to a binary classification target.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

from src.config import (
    DATASET_METADATA_PATH,
    EXPECTED_COLUMNS,
    FALLBACK_RAW_DATA_PATH,
    LABEL_MAPPING,
    ORIGINAL_CLASSES,
    RAW_DATA_PATH,
    TARGET_NAMES,
)


def find_dataset_path(custom_path: Optional[str] = None) -> Path:
    """Find the dataset file from custom path or standard locations."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified dataset file not found at: {custom_path}")

    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    if FALLBACK_RAW_DATA_PATH.exists():
        return FALLBACK_RAW_DATA_PATH

    raise FileNotFoundError(
        f"Dataset not found at '{RAW_DATA_PATH}' or '{FALLBACK_RAW_DATA_PATH}'. "
        "Please ensure 'cyberbullying_tweets.csv' is placed in the project root or 'data/' folder."
    )


def load_raw_dataset(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """Load raw dataset and extract initial data statistics (Stage 1)."""
    dataset_file = find_dataset_path(file_path)
    df = pd.read_csv(dataset_file)

    # Validate schema
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. Found columns: {list(df.columns)}"
        )

    # Calculate raw statistics
    raw_stats = {
        "file_path": str(dataset_file),
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_values": {k: int(v) for k, v in df.isnull().sum().items()},
        "exact_duplicates": int(df.duplicated().sum()),
        "duplicate_text_only": int(df.duplicated(subset=["tweet_text"]).sum()),
        "original_class_counts": {
            str(k): int(v) for k, v in df["cyberbullying_type"].value_counts().items()
        },
    }

    return df, raw_stats


def clean_and_prepare_dataset(
    df: pd.DataFrame, raw_stats: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """Clean data and prepare binary target (Stages 2 & 3).

    Cleaning rules:
    1. Drop rows with null or whitespace-only text or label.
    2. Filter out rows with unexpected labels not in ORIGINAL_CLASSES.
    3. Drop exact duplicates (identical tweet_text AND cyberbullying_type).
    4. For identical text with conflicting labels, drop to prevent label leakage/noise.
    5. Derive binary target (0: not_cyberbullying, 1: cyberbullying-related).
    """
    initial_count = len(df)
    clean_df = df.copy()

    # 1. Handle missing or non-string values
    clean_df = clean_df.dropna(subset=["tweet_text", "cyberbullying_type"])
    clean_df["tweet_text"] = clean_df["tweet_text"].astype(str)
    clean_df["cyberbullying_type"] = clean_df["cyberbullying_type"].astype(str).str.strip()

    # Drop empty or whitespace-only strings
    clean_df = clean_df[clean_df["tweet_text"].str.strip().str.len() > 0]
    after_empty_count = len(clean_df)
    empty_records_removed = initial_count - after_empty_count

    # 2. Filter valid classes
    clean_df = clean_df[clean_df["cyberbullying_type"].isin(ORIGINAL_CLASSES)]
    unexpected_classes_removed = after_empty_count - len(clean_df)

    # 3. Drop exact duplicates
    before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["tweet_text", "cyberbullying_type"])
    exact_duplicates_removed = before_dedup - len(clean_df)

    # 4. Handle identical text assigned conflicting labels (leakage and ambiguity prevention)
    before_conflict = len(clean_df)
    conflicting_texts = clean_df.groupby("tweet_text")["cyberbullying_type"].nunique()
    conflicting_texts = conflicting_texts[conflicting_texts > 1].index
    clean_df = clean_df[~clean_df["tweet_text"].isin(conflicting_texts)]
    conflicts_removed = before_conflict - len(clean_df)

    # 5. Target mapping (Stage 3)
    clean_df["target"] = clean_df["cyberbullying_type"].map(LABEL_MAPPING)
    clean_df["target"] = clean_df["target"].astype(int)
    clean_df["target_name"] = clean_df["target"].map(TARGET_NAMES)

    # Calculate text length stats
    clean_df["char_length"] = clean_df["tweet_text"].apply(len)
    clean_df["word_count"] = clean_df["tweet_text"].apply(lambda t: len(t.split()))

    final_count = len(clean_df)
    total_removed = initial_count - final_count

    # Compile comprehensive metadata
    metadata = {
        "raw_stats": raw_stats or {},
        "cleaning_summary": {
            "initial_records": initial_count,
            "empty_records_removed": empty_records_removed,
            "unexpected_classes_removed": unexpected_classes_removed,
            "exact_duplicates_removed": exact_duplicates_removed,
            "conflicting_duplicates_removed": conflicts_removed,
            "total_records_removed": total_removed,
            "final_records_retained": final_count,
        },
        "original_class_distribution": {
            str(k): int(v)
            for k, v in clean_df["cyberbullying_type"].value_counts().items()
        },
        "binary_target_distribution": {
            int(k): int(v) for k, v in clean_df["target"].value_counts().items()
        },
        "binary_target_proportions": {
            TARGET_NAMES[int(k)]: round(float(v) / final_count, 4)
            for k, v in clean_df["target"].value_counts().items()
        },
        "text_length_summary": {
            "char_length_mean": round(float(clean_df["char_length"].mean()), 2),
            "char_length_median": float(clean_df["char_length"].median()),
            "char_length_std": round(float(clean_df["char_length"].std()), 2),
            "word_count_mean": round(float(clean_df["word_count"].mean()), 2),
            "word_count_median": float(clean_df["word_count"].median()),
        },
    }

    return clean_df, metadata


def save_dataset_metadata(metadata: Dict, filepath: Optional[Path] = None) -> None:
    """Save dataset metadata to JSON."""
    out_path = filepath or DATASET_METADATA_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    df_raw, raw_stats = load_raw_dataset()
    print("--- RAW DATASET STATISTICS ---")
    print(f"Total records: {raw_stats['total_records']}")
    print(f"Original class counts: {raw_stats['original_class_counts']}")
    print(f"Exact duplicates: {raw_stats['exact_duplicates']}")

    clean_df, metadata = clean_and_prepare_dataset(df_raw, raw_stats)
    save_dataset_metadata(metadata)
    print("\n--- CLEANED DATASET SUMMARY ---")
    print(f"Final records retained: {metadata['cleaning_summary']['final_records_retained']}")
    print(f"Total removed: {metadata['cleaning_summary']['total_records_removed']}")
    print(f"Binary distribution: {metadata['binary_target_distribution']}")
    print(f"Saved metadata to {DATASET_METADATA_PATH}")
