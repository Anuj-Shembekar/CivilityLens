"""Training and Experimentation Pipeline for CivilityLens.

Executes:
- Stage 8: BoW and TF-IDF Feature Extraction
- Stage 9: Strict 70/15/15 Train-Validation-Test Splitting (Preventing Leakage)
- Stage 10: Training Multinomial Naive Bayes, Logistic Regression, and Linear SVM
- Section 7: Optional Controlled NLP Experiments (N-grams, Preprocessing, Char N-grams)
- Section 5: Model Selection on Validation set, followed by untouched Test Set Evaluation.
"""

import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    DATASET_METADATA_PATH,
    DEFAULT_BOW_CONFIG,
    DEFAULT_TFIDF_CONFIG,
    ERROR_ANALYSIS_PATH,
    EXPERIMENT_RESULTS_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    SUBTYPE_MODEL_PATH,
    TARGET_NAMES,
    TEST_RATIO,
    VAL_RATIO,
)
from src.data_loader import clean_and_prepare_dataset, load_raw_dataset, save_dataset_metadata
from src.evaluate import compute_classification_metrics, extract_error_analysis_cases
from src.preprocessing import (
    apply_lemmatization,
    apply_stemming,
    clean_text_conservative,
    get_custom_stopwords,
)


def split_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into 70% Train, 15% Validation, and 15% Test.

    Uses stratified splitting on the binary target to ensure consistent class
    ratios across splits.
    """
    # First split off Test (15%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    # Next split Validation (15% of total => 0.15 / 0.85 approx 0.1765 of train_val)
    val_relative_ratio = VAL_RATIO / (1.0 - TEST_RATIO)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_ratio,
        random_state=RANDOM_STATE,
        stratify=train_val_df["target"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def create_vectorizer(
    vec_type: str,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    analyzer: str = "word",
    use_stopwords: bool = False,
):
    """Instantiate a Scikit-Learn vectorizer based on experiment parameters."""
    stop_words = get_custom_stopwords() if use_stopwords else None

    if vec_type == "tfidf":
        return TfidfVectorizer(
            preprocessor=clean_text_conservative,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=2,
            sublinear_tf=True,
            analyzer=analyzer,
            stop_words=stop_words,
        )
    elif vec_type == "bow":
        return CountVectorizer(
            preprocessor=clean_text_conservative,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=2,
            analyzer=analyzer,
            stop_words=stop_words,
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {vec_type}")


def get_classifier(clf_name: str):
    """Instantiate a Scikit-Learn classifier with fixed random seeds."""
    if clf_name == "naive_bayes":
        return MultinomialNB(alpha=1.0)
    elif clf_name == "logistic_regression":
        return LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE, solver="liblinear"
        )
    elif clf_name == "linear_svm":
        return LinearSVC(
            C=1.0, random_state=RANDOM_STATE, max_iter=2000, dual="auto"
        )
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")


def run_experiment_suite(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """Train all required baseline models and controlled NLP experiments.

    Evaluates each configuration strictly on the Validation set.
    """
    experiments = [
        # Baseline Comparisons across 3 Classifiers with TF-IDF (1,2)
        {
            "experiment_id": "EXP-1A",
            "model_name": "Multinomial Naive Bayes",
            "clf_key": "naive_bayes",
            "vectorizer": "TF-IDF (1,2)",
            "vec_key": "tfidf",
            "ngrams": (1, 2),
            "analyzer": "word",
            "stopwords": False,
            "preprocessing": "Conservative Baseline",
        },
        {
            "experiment_id": "EXP-1B",
            "model_name": "Logistic Regression",
            "clf_key": "logistic_regression",
            "vectorizer": "TF-IDF (1,2)",
            "vec_key": "tfidf",
            "ngrams": (1, 2),
            "analyzer": "word",
            "stopwords": False,
            "preprocessing": "Conservative Baseline",
        },
        {
            "experiment_id": "EXP-1C",
            "model_name": "Linear SVM",
            "clf_key": "linear_svm",
            "vectorizer": "TF-IDF (1,2)",
            "vec_key": "tfidf",
            "ngrams": (1, 2),
            "analyzer": "word",
            "stopwords": False,
            "preprocessing": "Conservative Baseline",
        },
        # Bag of Words Comparisons
        {
            "experiment_id": "EXP-2A",
            "model_name": "Multinomial Naive Bayes",
            "clf_key": "naive_bayes",
            "vectorizer": "Bag of Words (1,2)",
            "vec_key": "bow",
            "ngrams": (1, 2),
            "analyzer": "word",
            "stopwords": False,
            "preprocessing": "Conservative Baseline",
        },
        {
            "experiment_id": "EXP-2B",
            "model_name": "Logistic Regression",
            "clf_key": "logistic_regression",
            "vectorizer": "Bag of Words (1,2)",
            "vec_key": "bow",
            "ngrams": (1, 2),
            "analyzer": "word",
            "stopwords": False,
            "preprocessing": "Conservative Baseline",
        },
        {
            "experiment_id": "EXP-2C",
            "model_name": "Linear SVM",
            "clf_key": "linear_svm",
            "vectorizer": "Bag of Words (1,2)",
            "vec_key": "bow",
            "ngrams": (1, 2),
            "analyzer": "word",
            "stopwords": False,
            "preprocessing": "Conservative Baseline",
        },
        # Unigram-Only Experiment (TF-IDF)
        {
            "experiment_id": "EXP-3",
            "model_name": "Logistic Regression",
            "clf_key": "logistic_regression",
            "vectorizer": "TF-IDF (Unigram only)",
            "vec_key": "tfidf",
            "ngrams": (1, 1),
            "analyzer": "word",
            "stopwords": False,
            "preprocessing": "Conservative Baseline",
        },
        # Stopword Removal Experiment (Negation-Preserving)
        {
            "experiment_id": "EXP-4",
            "model_name": "Logistic Regression",
            "clf_key": "logistic_regression",
            "vectorizer": "TF-IDF (1,2) + Stopwords",
            "vec_key": "tfidf",
            "ngrams": (1, 2),
            "analyzer": "word",
            "stopwords": True,
            "preprocessing": "Custom Stopword Removal",
        },
        # Character N-gram Experiment (Capturing obfuscated words)
        {
            "experiment_id": "EXP-5",
            "model_name": "Logistic Regression",
            "clf_key": "logistic_regression",
            "vectorizer": "TF-IDF Char N-grams (3,5)",
            "vec_key": "tfidf",
            "ngrams": (3, 5),
            "analyzer": "char_wb",
            "stopwords": False,
            "preprocessing": "Character-level N-grams",
        },
    ]

    results_list = []
    trained_pipelines = {}

    X_train = train_df["tweet_text"]
    y_train = train_df["target"].values
    X_val = val_df["tweet_text"]
    y_val = val_df["target"].values

    print(f"\nTraining and validating {len(experiments)} experimental configurations...")

    for exp in experiments:
        t0 = time.time()
        vec = create_vectorizer(
            vec_type=exp["vec_key"],
            ngram_range=exp["ngrams"],
            analyzer=exp["analyzer"],
            use_stopwords=exp["stopwords"],
        )
        clf = get_classifier(exp["clf_key"])

        pipeline = Pipeline([("vectorizer", vec), ("classifier", clf)])

        # Fit on training set only!
        pipeline.fit(X_train, y_train)

        # Predict on validation set
        val_preds = pipeline.predict(X_val)
        train_time = round(time.time() - t0, 2)

        # Compute metrics
        val_metrics = compute_classification_metrics(y_val, val_preds)

        exp_result = {
            "Experiment ID": exp["experiment_id"],
            "Model": exp["model_name"],
            "Vectorizer": exp["vectorizer"],
            "Preprocessing": exp["preprocessing"],
            "Val Accuracy": val_metrics["accuracy"],
            "Val Precision": val_metrics["precision"],
            "Val Recall": val_metrics["recall"],
            "Val F1-Score": val_metrics["f1_score"],
            "Val Macro F1": val_metrics["macro_f1"],
            "Training Time (s)": train_time,
        }
        results_list.append(exp_result)
        trained_pipelines[exp["experiment_id"]] = {
            "pipeline": pipeline,
            "config": exp,
            "val_metrics": val_metrics,
        }
        print(
            f"  [{exp['experiment_id']}] {exp['model_name']} with {exp['vectorizer']} -> "
            f"Val F1: {val_metrics['f1_score']:.4f}, Accuracy: {val_metrics['accuracy']:.4f} ({train_time}s)"
        )

    results_df = pd.DataFrame(results_list)
    return results_df, trained_pipelines


def train_and_evaluate_all():
    """Main training workflow executing complete pipeline and saving artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CIVILITYLENS: TRAINING & MODEL COMPARISON PIPELINE")
    print("=" * 70)

    # 1. Load raw data
    print("\n[Stage 1] Loading raw dataset...")
    df_raw, raw_stats = load_raw_dataset()

    # 2. Clean & target prep
    print("[Stage 2 & 3] Cleaning dataset and mapping binary target...")
    clean_df, metadata = clean_and_prepare_dataset(df_raw, raw_stats)
    save_dataset_metadata(metadata)

    # 3. Splitting
    print("[Stage 9] Splitting into 70% Train, 15% Validation, 15% Test...")
    train_df, val_df, test_df = split_dataset(clean_df)
    print(f"  Training set:   {len(train_df):,} samples")
    print(f"  Validation set: {len(val_df):,} samples")
    print(f"  Test set:       {len(test_df):,} samples")

    # 4. Run experiments and compare models on Validation set
    print("\n[Stages 8 & 10] Running model experiments on Validation set...")
    comparison_df, trained_pipelines = run_experiment_suite(train_df, val_df)

    # Save validation comparison table
    comparison_df.to_csv(EXPERIMENT_RESULTS_PATH, index=False)
    print(f"\nSaved experiment comparison to {EXPERIMENT_RESULTS_PATH}")

    # Select best model based on Validation F1-score
    best_exp_row = comparison_df.sort_values(
        by=["Val F1-Score", "Val Accuracy"], ascending=False
    ).iloc[0]
    best_exp_id = best_exp_row["Experiment ID"]
    best_pipeline_info = trained_pipelines[best_exp_id]
    best_pipeline = best_pipeline_info["pipeline"]

    print("\n" + "=" * 70)
    print(f"BEST MODEL SELECTED (Validation F1 = {best_exp_row['Val F1-Score']:.4f}):")
    print(f"Model: {best_exp_row['Model']} | Vectorizer: {best_exp_row['Vectorizer']}")
    print("=" * 70)

    # 5. UNTOUCHED FINAL TEST SET EVALUATION
    print("\n[Stage 6] Evaluating selected best model on UNTOUCHED Test Set...")
    X_test = test_df["tweet_text"]
    y_test = test_df["target"].values
    test_preds = best_pipeline.predict(X_test)

    # If model supports decision function or probabilities, compute decision info
    test_metrics = compute_classification_metrics(y_test, test_preds)

    print(f"\n--- FINAL UNTOUCHED TEST SET RESULTS ---")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    # 6. Extract Error Analysis Cases from Test Set
    print("\nExtracting representative error analysis cases (FP & FN)...")
    error_cases = extract_error_analysis_cases(
        texts=test_df["tweet_text"].tolist(),
        original_types=test_df["cyberbullying_type"].tolist(),
        y_true=y_test.tolist(),
        y_pred=test_preds.tolist(),
        max_cases_per_type=20,
    )
    with open(ERROR_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(error_cases, f, indent=2)
    print(f"Saved {len(error_cases['false_positives'])} FP and {len(error_cases['false_negatives'])} FN cases to {ERROR_ANALYSIS_PATH}")

    # 7. Train Stage 2 Multi-Class Subtype Classifier (Hierarchical Routing)
    print("\n[Stage 2 Hierarchical] Training Subtype Classifier (5 Cyberbullying Categories)...")
    from sklearn.metrics import accuracy_score
    train_cb = train_df[train_df["target"] == 1]
    val_cb = val_df[val_df["target"] == 1]
    test_cb = test_df[test_df["target"] == 1]

    subtype_pipeline = Pipeline([
        (
            "vec",
            TfidfVectorizer(
                preprocessor=clean_text_conservative,
                ngram_range=(1, 2),
                max_features=25000,
                sublinear_tf=True,
            ),
        ),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    subtype_pipeline.fit(train_cb["tweet_text"], train_cb["cyberbullying_type"])

    val_subtype_preds = subtype_pipeline.predict(val_cb["tweet_text"])
    test_subtype_preds = subtype_pipeline.predict(test_cb["tweet_text"])

    val_subtype_acc = float(accuracy_score(val_cb["cyberbullying_type"], val_subtype_preds))
    test_subtype_acc = float(accuracy_score(test_cb["cyberbullying_type"], test_subtype_preds))
    print(f"  Stage 2 Subtype Validation Accuracy: {val_subtype_acc*100:.2f}%")
    print(f"  Stage 2 Subtype Test Accuracy:       {test_subtype_acc*100:.2f}%")

    joblib.dump(subtype_pipeline, SUBTYPE_MODEL_PATH)
    print(f"Saved Stage 2 Subtype pipeline artifact to {SUBTYPE_MODEL_PATH}")

    # 8. Save Best Model Pipeline and Full Evaluation Metadata
    print("\nSaving best model pipeline and evaluation report...")
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"Saved trained pipeline artifact to {BEST_MODEL_PATH}")

    full_evaluation = {
        "best_model_id": best_exp_id,
        "best_model_name": best_exp_row["Model"],
        "best_vectorizer": best_exp_row["Vectorizer"],
        "preprocessing": best_exp_row["Preprocessing"],
        "validation_metrics": best_pipeline_info["val_metrics"],
        "final_test_metrics": test_metrics,
        "stage2_subtype_metrics": {
            "validation_accuracy": round(val_subtype_acc, 4),
            "test_accuracy": round(test_subtype_acc, 4),
            "categories": ["age", "ethnicity", "gender", "other_cyberbullying", "religion"],
        },
        "dataset_split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total_cleaned": len(clean_df),
        },
        "all_experiments": comparison_df.to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_evaluation, f, indent=2)
    print(f"Saved complete evaluation report to {METRICS_PATH}")
    print("\nPipeline training and evaluation completed successfully!")


if __name__ == "__main__":
    train_and_evaluate_all()
