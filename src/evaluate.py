"""Evaluation and Metric Utilities for CivilityLens.

Calculates Accuracy, Precision, Recall, F1-score, Confusion Matrix,
and extracts representative error analysis samples (False Positives and False Negatives).
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import TARGET_NAMES


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics for binary cyberbullying detection.

    Positive Class (1): Cyberbullying-related
    Negative Class (0): Non-cyberbullying
    """
    accuracy = float(accuracy_score(y_true, y_pred))

    # Binary metrics (pos_label=1)
    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision_pos), 4),
        "recall": round(float(recall_pos), 4),
        "f1_score": round(float(f1_pos), 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
        "per_class": {
            "non_cyberbullying (0)": {
                "precision": round(float(per_class_p[0]), 4),
                "recall": round(float(per_class_r[0]), 4),
                "f1_score": round(float(per_class_f1[0]), 4),
                "support": int(per_class_support[0]),
            },
            "cyberbullying_related (1)": {
                "precision": round(float(per_class_p[1]), 4),
                "recall": round(float(per_class_r[1]), 4),
                "f1_score": round(float(per_class_f1[1]), 4),
                "support": int(per_class_support[1]),
            },
        },
    }


def extract_error_analysis_cases(
    texts: List[str],
    original_types: List[str],
    y_true: List[int],
    y_pred: List[int],
    max_cases_per_type: int = 15,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract concrete False Positive and False Negative cases from evaluation predictions.

    False Positive: Non-cyberbullying (0) predicted as Cyberbullying-related (1).
    False Negative: Cyberbullying-related (1) predicted as Non-cyberbullying (0).
    """
    false_positives = []
    false_negatives = []

    for text, orig_type, true_label, pred_label in zip(
        texts, original_types, y_true, y_pred
    ):
        case = {
            "text": text,
            "original_category": orig_type,
            "true_label": int(true_label),
            "true_name": TARGET_NAMES[int(true_label)],
            "pred_label": int(pred_label),
            "pred_name": TARGET_NAMES[int(pred_label)],
        }

        if true_label == 0 and pred_label == 1:
            if len(false_positives) < max_cases_per_type:
                case["error_type"] = "False Positive"
                case["explanation"] = (
                    "Harmless text incorrectly flagged as cyberbullying. Often caused by "
                    "innocuous mention of sensitive terms, strong vocabulary without malicious intent, "
                    "or reclaimed slang."
                )
                false_positives.append(case)

        elif true_label == 1 and pred_label == 0:
            if len(false_negatives) < max_cases_per_type:
                case["error_type"] = "False Negative"
                case["explanation"] = (
                    "Cyberbullying missed by the model. Often caused by implicit harassment, "
                    "sarcasm, subtle mocking, lack of explicit slurs, or obfuscated spelling."
                )
                false_negatives.append(case)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
