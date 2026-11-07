"""Prediction and Inference Module for CivilityLens.

Loads the saved Scikit-learn Pipeline artifacts:
- Stage 1: Binary Cyberbullying Triage (Cyberbullying vs Non-Cyberbullying)
- Stage 2: Subtype Classifier (Religion, Age, Gender, Ethnicity, General Harassment)

Validates inputs, runs 2-stage hierarchical prediction, computes calibrated probabilities,
and formats transparent moderation-support outputs.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np

from src.config import (
    BEST_MODEL_PATH,
    DECISION_THRESHOLD,
    SUBTYPE_DISPLAY_NAMES,
    SUBTYPE_MODEL_PATH,
    TARGET_NAMES,
)
from src.preprocessing import clean_text_conservative


class CivilityLensPredictor:
    """Hierarchical 2-stage inference engine for cyberbullying detection."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        subtype_model_path: Optional[Path] = None,
    ):
        self.model_path = model_path or BEST_MODEL_PATH
        self.subtype_model_path = subtype_model_path or SUBTYPE_MODEL_PATH
        self.pipeline = None
        self.subtype_pipeline = None
        self._load_models()

    def _load_models(self):
        """Load trained stage 1 and stage 2 pipelines from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Stage 1 model artifact not found at {self.model_path}. "
                "Please train the model first by running 'python src/train.py'."
            )
        self.pipeline = joblib.load(self.model_path)

        if self.subtype_model_path.exists():
            try:
                self.subtype_pipeline = joblib.load(self.subtype_model_path)
            except Exception:
                self.subtype_pipeline = None

    @property
    def model_name(self) -> str:
        """Name of the classifier inside the primary pipeline."""
        if self.pipeline and "classifier" in self.pipeline.named_steps:
            return self.pipeline.named_steps["classifier"].__class__.__name__
        return "Unknown Model"

    @property
    def vectorizer_name(self) -> str:
        """Name of the vectorizer inside the primary pipeline."""
        if self.pipeline and "vectorizer" in self.pipeline.named_steps:
            return self.pipeline.named_steps["vectorizer"].__class__.__name__
        return "Unknown Vectorizer"

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """Perform hierarchical 2-stage cyberbullying classification.

        Stage 1: Detect whether text is Cyberbullying-related.
        Stage 2 (if Cyberbullying): Identify subtype (religion, age, gender, ethnicity, general).

        Returns structured diagnostic details.
        """
        if self.pipeline is None:
            self._load_models()

        # Input validation
        if not isinstance(raw_text, str) or not raw_text.strip():
            return {
                "status": "empty_input",
                "message": "Input text is empty. Please enter text to analyze.",
                "raw_text": raw_text,
                "cleaned_text": "",
                "prediction": None,
                "prediction_label": None,
                "headline": "No input provided.",
                "subtype_name": None,
            }

        cleaned_text = clean_text_conservative(raw_text)

        # Stage 1: Primary Binary Classification
        pred_label = int(self.pipeline.predict([cleaned_text])[0])
        pred_name = TARGET_NAMES.get(pred_label, "Unknown")
        is_cyberbullying = bool(pred_label == 1)

        classifier = self.pipeline.named_steps["classifier"]
        vectorizer = self.pipeline.named_steps["vectorizer"]

        # Probability / Decision Score Handling
        score_type = "unknown"
        score_value = None
        cyberbullying_probability = None
        decision_score = None
        confidence_text = ""

        if hasattr(classifier, "predict_proba"):
            try:
                probs = self.pipeline.predict_proba([cleaned_text])[0]
                prob_neg = float(probs[0])
                prob_pos = float(probs[1])
                score_type = "calibrated_probability"
                cyberbullying_probability = prob_pos

                # Apply calibrated threshold (0.65) to eliminate false alarms from the 83% positive class prior
                is_cyberbullying = bool(cyberbullying_probability >= DECISION_THRESHOLD)
                pred_label = 1 if is_cyberbullying else 0
                pred_name = TARGET_NAMES.get(pred_label, "Unknown")

                score_value = prob_pos if is_cyberbullying else prob_neg
                confidence_text = f"{score_value * 100:.1f}%"
            except Exception:
                pass
        elif hasattr(classifier, "decision_function"):
            try:
                raw_decision = float(classifier.decision_function(vectorizer.transform([cleaned_text]))[0])
                score_type = "svm_decision_score"
                decision_score = round(raw_decision, 4)
                score_value = decision_score
                confidence_text = f"Decision Score: {decision_score:+.3f}"
            except Exception:
                pass

        # Risk assessment
        if is_cyberbullying:
            if cyberbullying_probability is not None and cyberbullying_probability >= 0.85:
                risk_level = "High Concern"
            elif cyberbullying_probability is not None and cyberbullying_probability >= 0.65:
                risk_level = "Moderate Concern"
            else:
                risk_level = "Flagged for Review"
        else:
            if cyberbullying_probability is not None and cyberbullying_probability < 0.25:
                risk_level = "Low Concern"
            else:
                risk_level = "Neutral / Non-flagged"

        # Stage 2: Subtype Identification (if Cyberbullying)
        subtype_key = None
        subtype_name = None
        subtype_confidence = None
        subtype_breakdown = {}
        headline = "Non-cyberbullying content."

        if is_cyberbullying:
            if self.subtype_pipeline is not None:
                try:
                    subtype_pred = self.subtype_pipeline.predict([cleaned_text])[0]
                    subtype_key = str(subtype_pred)
                    subtype_name = SUBTYPE_DISPLAY_NAMES.get(subtype_key, subtype_key.title())

                    # Subtype probabilities
                    if hasattr(self.subtype_pipeline.named_steps["clf"], "predict_proba"):
                        s_probs = self.subtype_pipeline.predict_proba([cleaned_text])[0]
                        classes = self.subtype_pipeline.named_steps["clf"].classes_
                        for c, p in zip(classes, s_probs):
                            display_k = SUBTYPE_DISPLAY_NAMES.get(str(c), str(c).title())
                            subtype_breakdown[display_k] = round(float(p) * 100, 1)

                        top_prob = float(s_probs[list(classes).index(subtype_pred)])
                        subtype_confidence = round(top_prob * 100, 1)

                    # Formulate requested user headline
                    headline = f"Cyberbullying detected. Possibly {subtype_name.lower()}."
                except Exception:
                    headline = "Cyberbullying detected."
            else:
                headline = "Cyberbullying detected."
        else:
            headline = "Non-cyberbullying content."

        return {
            "status": "success",
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "prediction": pred_label,
            "prediction_label": pred_name,
            "is_cyberbullying": is_cyberbullying,
            "risk_level": risk_level,
            "headline": headline,
            "subtype_key": subtype_key,
            "subtype_name": subtype_name,
            "subtype_confidence": subtype_confidence,
            "subtype_probabilities": subtype_breakdown,
            "score_type": score_type,
            "score_value": score_value,
            "cyberbullying_probability": cyberbullying_probability,
            "decision_score": decision_score,
            "confidence_display": confidence_text,
            "model_architecture": f"{self.vectorizer_name} + {self.model_name}",
            "disclaimer": (
                "CivilityLens is an assistive moderation-support tool, not an automated judge. "
                "Subtype predictions indicate statistical semantic alignment and require human contextual review."
            ),
        }


# Global singleton instance
_cached_predictor: Optional[CivilityLensPredictor] = None


def get_predictor() -> CivilityLensPredictor:
    """Get or initialize singleton CivilityLensPredictor."""
    global _cached_predictor
    if _cached_predictor is None:
        _cached_predictor = CivilityLensPredictor()
    return _cached_predictor
