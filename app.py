"""CivilityLens - Modern NLP Cyberbullying Detection Application.

Streamlit Moderation Support Application powered by Scikit-Learn NLP Pipeline.
Displays real dataset statistics, actual trained model metrics,
interactive pipeline visualization, and live text classification.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

# Application Configuration & Custom Styling
st.set_page_config(
    page_title="CivilityLens | NLP Cyberbullying Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Charcoal & Teal Theme
st.markdown(
    """
    <style>
    /* Global Theme Styles */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .hero-banner {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        border-left: 5px solid #38bdf8;
        padding: 24px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    .tagline {
        color: #38bdf8;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 5px;
    }
    .disclaimer-box {
        background-color: #1c2128;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 20px;
        font-size: 0.9rem;
        color: #e3b341;
    }
    .result-flagged {
        background: linear-gradient(135deg, #3b1114 0%, #22080a 100%);
        border: 1px solid #f85149;
        border-radius: 10px;
        padding: 24px;
        margin-top: 20px;
    }
    .result-safe {
        background: linear-gradient(135deg, #0d2818 0%, #05140b 100%);
        border: 1px solid #3fb950;
        border-radius: 10px;
        padding: 24px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = ARTIFACTS_DIR / "best_model_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
COMPARISON_PATH = ARTIFACTS_DIR / "experiment_comparison.csv"
METADATA_PATH = ARTIFACTS_DIR / "dataset_metadata.json"
ERROR_SAMPLES_PATH = ARTIFACTS_DIR / "error_analysis_samples.json"


@st.cache_data
def load_cached_artifacts():
    """Load precomputed metrics, metadata, and comparisons from disk."""
    metadata = {}
    metrics = {}
    comparison_df = pd.DataFrame()
    error_samples = {}

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    if COMPARISON_PATH.exists():
        comparison_df = pd.read_csv(COMPARISON_PATH)

    if ERROR_SAMPLES_PATH.exists():
        with open(ERROR_SAMPLES_PATH, "r", encoding="utf-8") as f:
            error_samples = json.load(f)

    return metadata, metrics, comparison_df, error_samples


metadata, metrics, comparison_df, error_samples = load_cached_artifacts()


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("CivilityLens")
st.sidebar.caption("Assistive NLP Cyberbullying Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏛️ Overview",
        "📊 Dataset Explorer",
        "⚙️ NLP Pipeline Architecture",
        "🔬 Model Laboratory",
        "⚡ Live Text Detection",
        "🔍 Error Analysis & Failure Modes",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Academic Project Details**
    - **Focus:** NLP Text Classification
    - **Dataset:** Kaggle Cyberbullying (Andrew M.)
    - **Pipeline:** Preprocessing → TF-IDF → Linear Model
    - **Target:** Binary Moderation Support
    """
)

# -----------------------------------------------------------------------------
# PAGE A: OVERVIEW
# -----------------------------------------------------------------------------
if page == "🏛️ Overview":
    st.markdown(
        """
        <div class="hero-banner">
            <h1 style="margin: 0; color: #ffffff;">CivilityLens</h1>
            <div class="tagline">Detect harmful patterns. Reduce false alarms. Support safer online communities.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Executive Summary
        **CivilityLens** is an academic Natural Language Processing and Machine Learning application designed to assist human moderators in detecting online harassment and cyberbullying patterns.
        
        Rather than acting as an infallible automated judge, CivilityLens functions as a **moderation-support tool** that flags potentially abusive content, provides transparent probability estimates, and explains the linguistic features that drove the decision.
        """
    )

    # Key Performance Indicators
    st.subheader("Measured Test-Set Performance")
    if metrics and "final_test_metrics" in metrics:
        test_m = metrics["final_test_metrics"]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Test Accuracy</div>
                    <div class="metric-value">{test_m['accuracy'] * 100:.1f}%</div>
                    <div style="color: #8b949e; font-size: 0.8rem;">Evaluated on 6,657 records</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Test Precision</div>
                    <div class="metric-value">{test_m['precision'] * 100:.1f}%</div>
                    <div style="color: #8b949e; font-size: 0.8rem;">Positive class reliability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Test Recall</div>
                    <div class="metric-value">{test_m['recall'] * 100:.1f}%</div>
                    <div style="color: #8b949e; font-size: 0.8rem;">Harassment pattern coverage</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Test F1-Score</div>
                    <div class="metric-value">{test_m['f1_score']:.4f}</div>
                    <div style="color: #8b949e; font-size: 0.8rem;">Harmonic mean (P & R)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("Model evaluation metrics not found. Run `python src/train.py` to generate real metrics.")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Model Status")
        if MODEL_PATH.exists():
            st.success(f"Active Trained Model: **{metrics.get('best_model_name', 'Logistic Regression')}**")
            st.info(f"Feature Extraction: **{metrics.get('best_vectorizer', 'TF-IDF Char N-grams (3,5)')}**")
            st.write(
                f"Selected automatically based on highest Validation F1-score ({metrics.get('validation_metrics', {}).get('f1_score', 'N/A')})."
            )
        else:
            st.error("Model artifact not found. Please train the model using `python src/train.py`.")

    with col_right:
        st.subheader("Dataset Verification")
        if metadata and "cleaning_summary" in metadata:
            raw_c = metadata["cleaning_summary"]["initial_records"]
            retained_c = metadata["cleaning_summary"]["final_records_retained"]
            st.write(f"- **Raw Dataset Records:** `{raw_c:,}` rows")
            st.write(f"- **Cleaned Dataset Records:** `{retained_c:,}` rows")
            st.write(f"- **Original Classes:** 6 categories (Age, Ethnicity, Gender, Religion, Other, Not Cyberbullying)")
            st.write(f"- **Current Task:** Binary Classification (`0`: Non-cyberbullying, `1`: Cyberbullying-related)")
        else:
            st.info("Dataset statistics will populate after running data cleaning.")

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Product Principle & Academic Disclaimer:</strong><br>
            CivilityLens is designed to augment human judgment, not replace it. Text classification models analyze statistical token frequencies and cannot discern true intent, contextual sarcasm, or private nuances. No automated flag should be treated as an absolute moral judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# PAGE B: DATASET EXPLORER
# -----------------------------------------------------------------------------
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer & Exploratory Analysis")
    st.write(
        "Inspect the Kaggle Cyberbullying Classification dataset used for training, "
        "including raw distributions, cleaning rules, and text length statistics."
    )

    if metadata and "raw_stats" in metadata:
        col1, col2, col3, col4 = st.columns(4)
        raw_s = metadata["raw_stats"]
        clean_s = metadata["cleaning_summary"]

        col1.metric("Raw Samples", f"{raw_s.get('total_records', 0):,}")
        col2.metric("Exact Duplicates", f"{raw_s.get('exact_duplicates', 0):,}")
        col3.metric("Conflicting Duplicates Removed", f"{clean_s.get('conflicting_duplicates_removed', 0):,}")
        col4.metric("Retained Clean Samples", f"{clean_s.get('final_records_retained', 0):,}")

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["🏷️ Class Distributions", "📐 Text Length Metrics", "📋 Cleaning Decisions"])

        with tab1:
            st.subheader("Original Multi-Class Distribution")
            orig_dist = pd.DataFrame(
                list(metadata["original_class_distribution"].items()),
                columns=["Category", "Sample Count"],
            )
            st.bar_chart(orig_dist.set_index("Category"))

            st.subheader("Binary Target Mapping")
            st.markdown(
                """
                - **Class 0 (Non-cyberbullying):** `not_cyberbullying`
                - **Class 1 (Cyberbullying-related):** `religion`, `age`, `gender`, `ethnicity`, `other_cyberbullying`
                """
            )
            bin_dist = pd.DataFrame(
                [
                    {"Target": "Non-cyberbullying (0)", "Count": metadata["binary_target_distribution"].get("0", 0)},
                    {"Target": "Cyberbullying-related (1)", "Count": metadata["binary_target_distribution"].get("1", 0)},
                ]
            )
            st.dataframe(bin_dist, use_container_width=True)

        with tab2:
            st.subheader("Text Characteristics")
            length_s = metadata.get("text_length_summary", {})
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Mean Char Length", f"{length_s.get('char_length_mean', 0)} chars")
            col_b.metric("Median Char Length", f"{length_s.get('char_length_median', 0)} chars")
            col_c.metric("Mean Word Count", f"{length_s.get('word_count_mean', 0)} words")
            col_d.metric("Median Word Count", f"{length_s.get('word_count_median', 0)} words")

        with tab3:
            st.subheader("Data Cleaning & Leakage Prevention Rules")
            st.markdown(
                """
                1. **Exact Duplicate Removal:** Tweets with identical text and identical labels were deduplicated.
                2. **Conflicting Label Resolution:** Tweets with identical text assigned to conflicting labels were purged to prevent contradictory gradient updates and evaluation contamination.
                3. **Conservative Filtering:** Text with whitespace only or missing values was eliminated.
                4. **Splitting Integrity:** Splitting occurred *after* deduplication to prevent near-duplicate leakage between training and test sets.
                """
            )
    else:
        st.warning("Dataset metadata not found. Please run the training script.")

# -----------------------------------------------------------------------------
# PAGE C: NLP PIPELINE ARCHITECTURE
# -----------------------------------------------------------------------------
elif page == "⚙️ NLP Pipeline Architecture":
    st.title("⚙️ Complete NLP Pipeline Architecture")
    st.write("Understand the scientific rationale behind each step from raw social media text to final classification.")

    stages = [
        {
            "num": "Stage 1",
            "name": "Dataset Ingestion & Validation",
            "desc": "Validates incoming CSV schema (`tweet_text`, `cyberbullying_type`), checks for null values, and inspects data integrity.",
            "why": "Prevents downstream pipeline crashes caused by corrupted data types or missing fields.",
        },
        {
            "num": "Stage 2 & 3",
            "name": "Cleaning & Binary Target Derivation",
            "desc": "Purges exact and conflicting duplicates. Maps the 6 original classes to binary target (0: non-cyberbullying, 1: cyberbullying-related).",
            "why": "Simplifies initial modeling into high-recall moderation triage while preserving original multi-class labels for future work.",
        },
        {
            "num": "Stage 4",
            "name": "Conservative Text Normalisation",
            "desc": "Converts to lowercase, substitutes URLs with `urltoken`, replaces @mentions with `usertoken`, strips '#' while keeping hashtag text, and cleans repeated whitespace.",
            "why": "Avoids over-aggressive filtering. Retains punctuation and emojis which convey emotional aggression.",
        },
        {
            "num": "Stage 5 & 6",
            "name": "Tokenisation & Controlled Stopword Handling",
            "desc": "Segments text into token n-grams. Critically retains negation words ('not', 'no', 'never', 'without') to preserve polarity.",
            "why": "Deleting negations casually in sentiment/harassment tasks inverts sentence meaning (e.g. 'not ugly' vs 'ugly').",
        },
        {
            "num": "Stage 8",
            "name": "Vectorisation (TF-IDF vs Bag of Words)",
            "desc": "Transforms discrete tokens into continuous sparse feature vectors using Term Frequency-Inverse Document Frequency or Term Counts.",
            "why": "Down-weights ubiquitous generic terms and elevates discriminative terms characteristic of abusive attacks.",
        },
        {
            "num": "Stage 9",
            "name": "Strict Train / Val / Test Splitting",
            "desc": "70% Training, 15% Validation, 15% Test. Fitted vectorizers apply only to training data; test data remains completely untouched until final scoring.",
            "why": "Guarantees unbiased generalization metrics without data snooping or target leakage.",
        },
        {
            "num": "Stage 10",
            "name": "Model Classification & Calibration",
            "desc": "Compares Naive Bayes, Logistic Regression, and Linear SVM. Generates calibrated probabilities or hyperplane decision scores.",
            "why": "Enables threshold-aware decision making for moderation workflows.",
        },
    ]

    for s in stages:
        with st.expander(f"{s['num']} — {s['name']}", expanded=(s['num'] in ["Stage 4", "Stage 8"])):
            st.write(f"**What happens:** {s['desc']}")
            st.info(f"**Why it matters:** {s['why']}")

    st.markdown("---")
    st.subheader("Interactive Normalisation Sandbox")
    sample_input = st.text_input(
        "Enter raw text to view live pipeline normalisation:",
        value="RT @angry_user: Go away #pathetic loser!! Check out https://t.co/fake and never come back :(",
    )

    from src.preprocessing import clean_text_conservative, apply_stemming, apply_lemmatization

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Original Raw Text:**")
        st.code(sample_input, language="text")

        st.markdown("**Stage 4 Conservative Normalised Text:**")
        cleaned = clean_text_conservative(sample_input)
        st.code(cleaned, language="text")

    with col_b:
        st.markdown("**Stage 7 Porter Stemming:**")
        st.code(apply_stemming(sample_input), language="text")

        st.markdown("**Stage 7 WordNet Lemmatisation:**")
        st.code(apply_lemmatization(sample_input), language="text")

# -----------------------------------------------------------------------------
# PAGE D: MODEL LABORATORY
# -----------------------------------------------------------------------------
elif page == "🔬 Model Laboratory":
    st.title("🔬 Model Laboratory & Experimental Comparison")
    st.write(
        "Compare empirical results from all experimental configurations on the exact same validation splits."
    )

    if not comparison_df.empty:
        st.subheader("Model Comparison Table (Measured Validation Results)")
        st.dataframe(
            comparison_df.style.highlight_max(
                subset=["Val F1-Score", "Val Accuracy", "Val Precision", "Val Recall"],
                color="#1b4d3e",
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("Key Empirical Insights from the Lab")
        st.markdown(
            """
            1. **Character N-grams vs Word N-grams:** Character n-grams (3-5) outperformed standard word n-grams on validation F1. Character sequences capture deliberate obfuscation, leetspeak, and misspelled insults common in online harassment.
            2. **TF-IDF vs Bag of Words:** TF-IDF consistently yielded higher precision by dampening the influence of frequent non-informative words.
            3. **Linear Classifiers:** Both Logistic Regression and Linear SVM showed strong margins on high-dimensional sparse representations, outperforming Naive Bayes which assumes strict conditional feature independence.
            """
        )

        # Confusion Matrix Display
        if metrics and "final_test_metrics" in metrics:
            st.markdown("---")
            st.subheader("Final Untouched Test Set Evaluation")
            cm_dict = metrics["final_test_metrics"]["confusion_matrix"]
            tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]

            col_cm1, col_cm2 = st.columns(2)
            with col_cm1:
                st.markdown("**Confusion Matrix (6,657 Test Samples)**")
                cm_df = pd.DataFrame(
                    [
                        [f"TN: {tn:,}", f"FP: {fp:,}"],
                        [f"FN: {fn:,}", f"TP: {tp:,}"],
                    ],
                    index=["Actual Non-Bullying (0)", "Actual Bullying (1)"],
                    columns=["Predicted Non-Bullying (0)", "Predicted Bullying (1)"],
                )
                st.dataframe(cm_df, use_container_width=True)

            with col_cm2:
                st.markdown("**Error Distribution Breakdown**")
                st.write(f"- **True Negatives (TN):** `{tn:,}` — Harmless text correctly classified")
                st.write(f"- **True Positives (TP):** `{tp:,}` — Harassment correctly flagged")
                st.write(f"- **False Positives (FP):** `{fp:,}` — False alarms on benign messages")
                st.write(f"- **False Negatives (FN):** `{fn:,}` — Missed cyberbullying instances")
    else:
        st.warning("No experiment results found. Please run `python src/train.py`.")

# -----------------------------------------------------------------------------
# PAGE E: LIVE TEXT DETECTION
# -----------------------------------------------------------------------------
elif page == "⚡ Live Text Detection":
    st.title("⚡ Live Cyberbullying Detection")
    st.write(
        "Enter text into the moderation workbench to run real-time inference through the trained NLP pipeline."
    )

    from src.predict import get_predictor

    user_text = st.text_area(
        "Enter text to evaluate for cyberbullying patterns:",
        height=130,
        placeholder="Type or paste any tweet, comment, or message here...",
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze Text", type="primary")

    if analyze_clicked:
        if not user_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            try:
                predictor = get_predictor()
                result = predictor.predict(user_text)

                st.markdown("---")
                if result["is_cyberbullying"]:
                    subtype_heading = result.get("headline", "Cyberbullying detected.")
                    st.markdown(
                        f"""
                        <div class="result-flagged">
                            <h2 style="color: #f85149; margin-top: 0;">⚠️ {subtype_heading}</h2>
                            <p style="font-size: 1.1rem; color: #ff7b72; margin: 8px 0;">
                                <strong>Assessment:</strong> This text exhibits linguistic patterns frequently associated with online harassment.
                            </p>
                            <p style="color: #e6edf3; margin-top: 6px;">
                                <strong>Overall Confidence:</strong> {result['confidence_display']} | <strong>Risk Level:</strong> {result['risk_level']}
                                {f" | <strong>Subtype Confidence:</strong> {result['subtype_confidence']}%" if result.get('subtype_confidence') else ""}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Subtype Probability Chart
                    if result.get("subtype_probabilities"):
                        st.subheader("Subtype Probability Distribution (Stage 2 Hierarchical Router)")
                        sub_df = pd.DataFrame(
                            list(result["subtype_probabilities"].items()),
                            columns=["Subtype Category", "Likelihood (%)"],
                        ).sort_values(by="Likelihood (%)", ascending=False)
                        st.bar_chart(sub_df.set_index("Subtype Category"))
                else:
                    st.markdown(
                        f"""
                        <div class="result-safe">
                            <h2 style="color: #3fb950; margin-top: 0;">✅ Non-Cyberbullying Content</h2>
                            <p style="font-size: 1.1rem; color: #56d364;">
                                <strong>Assessment:</strong> No significant indicators of harassment were detected in this message.
                            </p>
                            <p style="color: #e6edf3;">
                                <strong>Confidence Estimate:</strong> {result['confidence_display']} | <strong>Risk Level:</strong> {result['risk_level']}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Diagnostic details
                st.subheader("Pipeline Diagnostic Trace")
                trace_df = pd.DataFrame(
                    [
                        {"Field": "Active Model", "Value": result["model_architecture"]},
                        {"Field": "Cleaned Normalised Form", "Value": result["cleaned_text"]},
                        {"Field": "Detected Subtype", "Value": str(result.get("subtype_name", "None"))},
                        {"Field": "Score Metric", "Value": result["score_type"]},
                        {"Field": "Score Value", "Value": str(result["score_value"])},
                    ]
                )
                st.dataframe(trace_df, use_container_width=True)

                st.markdown(
                    f"""
                    <div class="disclaimer-box">
                        <strong>🛡️ Moderation Note:</strong> {result['disclaimer']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Inference error: {e}")

# -----------------------------------------------------------------------------
# PAGE F: ERROR ANALYSIS & FAILURE MODES
# -----------------------------------------------------------------------------
elif page == "🔍 Error Analysis & Failure Modes":
    st.title("🔍 Error Analysis & Failure Modes")
    st.write(
        "Responsible machine learning requires examining where and why the system fails. "
        "Review actual misclassified test cases extracted during evaluation."
    )

    if error_samples and ("false_positives" in error_samples or "false_negatives" in error_samples):
        tab_fp, tab_fn, tab_crit = st.tabs(
            ["False Positives (False Alarms)", "False Negatives (Missed Harassment)", "Failure Taxonomy"]
        )

        with tab_fp:
            st.subheader("False Positives (Ground Truth: Non-Cyberbullying | Predicted: Cyberbullying)")
            st.write(
                "These cases represent benign communications that the statistical model incorrectly flagged. "
                "Such errors cause user frustration and over-moderation."
            )
            fps = error_samples.get("false_positives", [])
            for i, fp in enumerate(fps[:8], 1):
                with st.expander(f"Case FP-{i}: Original Category: {fp.get('original_category', 'not_cyberbullying')}"):
                    st.markdown(f"**Tweet Text:** `{fp['text']}`")
                    st.info(f"**Root Cause:** {fp['explanation']}")

        with tab_fn:
            st.subheader("False Negatives (Ground Truth: Cyberbullying | Predicted: Non-Cyberbullying)")
            st.write(
                "These cases represent genuine harassment that bypassed detection. "
                "Such errors leave victims unprotected."
            )
            fns = error_samples.get("false_negatives", [])
            for i, fn in enumerate(fns[:8], 1):
                with st.expander(f"Case FN-{i}: Original Category: {fn.get('original_category', 'cyberbullying')}"):
                    st.markdown(f"**Tweet Text:** `{fn['text']}`")
                    st.warning(f"**Root Cause:** {fn['explanation']}")

        with tab_crit:
            st.subheader("Linguistic Failure Taxonomy")
            st.markdown(
                """
                | Failure Mode | Description | Example Challenge |
                |---|---|---|
                | **Sarcasm & Irony** | Surface-level polite words masking hostile intent. | *"Oh wow, you must be a real genius to think that."* |
                | **Reclaimed Slang** | In-group community vocabulary that appears toxic out of context. | Friendly banter between peers using provocative slang. |
                | **Implicit Harassment** | Passive-aggressive exclusion without explicit vulgarity or slurs. | *"Nobody asked you to sit here anyway."* |
                | **Adversarial Obfuscation** | Intentional leetspeak or character substitution to bypass lexicons. | Spacing out letters or substituting numbers (`b!tch`, `l0ser`). |
                | **Cross-Platform Shift** | Distribution shift from Twitter formatting to Discord or gaming chats. | Gaming slang and rapid shorthand not represented in training tweets. |
                """
            )
    else:
        st.info("Error analysis samples will appear after model training (`python src/train.py`).")
