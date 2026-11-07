"""Unit tests for prediction, model loading, and evaluation in CivilityLens."""

import unittest
import numpy as np
from src.config import BEST_MODEL_PATH
from src.data_loader import find_dataset_path
from src.evaluate import compute_classification_metrics
from src.predict import CivilityLensPredictor, get_predictor
from src.preprocessing import clean_text_conservative


class TestPredictionAndEvaluation(unittest.TestCase):

    def test_compute_classification_metrics(self):
        """Verify classification metrics accurately compute binary and matrix statistics."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)

        # Accuracy: 3 / 5 = 0.60
        self.assertEqual(metrics["accuracy"], 0.60)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 2)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 1)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 1)

    def test_zero_division_safety(self):
        """Verify metrics handle edge cases where a class has 0 predictions without crashing."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1_score"], 0.0)

    def test_find_dataset_path_valid(self):
        """Verify find_dataset_path locates the CSV without raising error."""
        path = find_dataset_path()
        self.assertTrue(path.exists())

    def test_find_dataset_path_invalid(self):
        """Verify find_dataset_path raises FileNotFoundError for nonexistent paths."""
        with self.assertRaises(FileNotFoundError):
            find_dataset_path("non_existent_dataset_12345.csv")

    def test_predictor_model_loading_and_inference(self):
        """Verify saved pipeline loads and produces valid prediction dictionary."""
        self.assertTrue(BEST_MODEL_PATH.exists(), "Trained model pipeline must exist on disk.")

        predictor = get_predictor()
        self.assertIsNotNone(predictor.pipeline)

        # Test cyberbullying prediction with Stage 2 Subtype Headline
        sample_bullying = "You are pathetic and nobody likes you, go away loser"
        result_b = predictor.predict(sample_bullying)

        self.assertEqual(result_b["status"], "success")
        self.assertEqual(result_b["prediction"], 1)
        self.assertTrue(result_b["is_cyberbullying"])
        self.assertIn("headline", result_b)
        self.assertTrue("Cyberbullying detected" in result_b["headline"])
        self.assertIsNotNone(result_b["subtype_name"])
        self.assertIn("disclaimer", result_b)

        # Test benign text prediction
        sample_benign = "hello how are you today"
        result_n = predictor.predict(sample_benign)

        self.assertEqual(result_n["status"], "success")
        self.assertEqual(result_n["prediction"], 0)
        self.assertFalse(result_n["is_cyberbullying"])
        self.assertEqual(result_n["prediction_label"], "Non-cyberbullying")
        self.assertEqual(result_n["headline"], "Non-cyberbullying content.")
        self.assertIsNone(result_n["subtype_name"])

    def test_predictor_empty_input(self):
        """Verify predictor gracefully handles empty or whitespace input."""
        predictor = get_predictor()
        res = predictor.predict("   ")
        self.assertEqual(res["status"], "empty_input")
        self.assertIsNone(res["prediction"])


if __name__ == "__main__":
    unittest.main()
