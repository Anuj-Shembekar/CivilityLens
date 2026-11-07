# CivilityLens 🛡️
### Assistive NLP & Machine Learning Cyberbullying Detection Application

> **Tagline:** Detect harmful patterns. Reduce false alarms. Support safer online communities.  
> **Course:** BTech Computer Science — Natural Language Processing & Machine Learning Lab Project  
> **Architecture:** Reusable Conservative Normalisation → Sublinear TF-IDF Character N-grams → Calibrated Logistic Regression  
> **Evaluated Test F1-Score:** `0.9427` (on 6,657 untouched test records)

---

## 1. Project Overview

**CivilityLens** is an end-to-end Natural Language Processing (NLP) and machine-learning software application designed to detect cyberbullying and online harassment in social media text. 

Built from scratch as an academically defensible university project, CivilityLens demonstrates:
- A leak-free data engineering and deduplication pipeline on 47,692 tweets.
- Systematic empirical comparison across 9 experimental configurations (Naive Bayes, Logistic Regression, Linear SVM, Bag of Words, TF-IDF, character n-grams, and controlled stopword filtering).
- Model selection strictly conducted on a dedicated validation split, followed by a single evaluation on an untouched test set.
- A modern interactive moderation-support web interface.
- Concrete error analysis tracing false alarms (False Positives) and missed attacks (False Negatives).

### ⚠️ Product Principle
CivilityLens is a **moderation-support tool**, not an automated judge of a person's character or intent. It computes statistical linguistic affinities to triage moderation queues, leaving consequential decisions to human reviewers.

---

## 2. Problem Statement

With hundreds of millions of user-generated comments posted daily across social platforms, human moderators are overwhelmed by toxic volume. However, naive automated filters that rely on crude keyword blacklists suffer from:
1. **Severe False Positives:** Banning users for quoting harassment, using non-aggressive profanity, or speaking in dialect.
2. **Adversarial Evasion:** Failing to detect intentionally misspelled slurs, leetspeak (`b!tch`), or subtle passive-aggressive bullying.
3. **Black-Box Opacity:** Offering no confidence scores, risk categories, or failure mode diagnostics.

CivilityLens solves this by combining sub-word character representations with calibrated linear classifiers and transparent diagnostics.

---

## 3. Dataset Source & Setup

The project uses the Kaggle Cyberbullying Classification dataset by Andrew M.:
- **URL:** [https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)
- **Local File:** `data/cyberbullying_tweets.csv` (or in project root)
- **Raw Volume:** 47,692 rows

### Download Instructions:
1. Download `cyberbullying_tweets.csv` from Kaggle.
2. Place the file in the `data/` directory:
   ```bash
   mkdir -p data
   cp /path/to/downloaded/cyberbullying_tweets.csv data/cyberbullying_tweets.csv
   ```
*(The repository already comes configured with dataset auto-discovery in `data/` and root).*

---

## 4. Dataset Schema & Original Categories

The raw dataset contains two string columns:
- `tweet_text`: Raw Twitter message text.
- `cyberbullying_type`: Annotated category label.

### Distribution across 47,692 Raw Tweets:
| Category | Sample Count | Percentage | Description |
|---|---|---|---|
| `religion` | 7,998 | 16.77% | Faith-targeted hate and insults |
| `age` | 7,992 | 16.76% | Schoolyard, youthful, or elderly bullying |
| `gender` | 7,973 | 16.72% | Sexist slurs and misogynistic harassment |
| `ethnicity` | 7,961 | 16.69% | Racial and xenophobic abuse |
| `not_cyberbullying` | 7,945 | 16.66% | Benign daily tweets, TV discussion, sports |
| `other_cyberbullying` | 7,823 | 16.40% | General harassment without specific identity tag |

---

## 5. Binary Target Mapping

To provide actionable triage for frontline content moderation, the six classes are mapped into a binary target:

$$\text{Target} = \begin{cases} 0 & \text{if } \text{cyberbullying\_type} = \text{'not\_cyberbullying'} \\ 1 & \text{if } \text{cyberbullying\_type} \in \{\text{'religion'}, \text{'age'}, \text{'gender'}, \text{'ethnicity'}, \text{'other\_cyberbullying'}\} \end{cases}$$

- **Target 0 (Negative Class):** Non-cyberbullying (benign content).
- **Target 1 (Positive Class):** Cyberbullying-related content.
- Original labels are retained in metadata for post-hoc error analysis.

---

## 6. Complete NLP Pipeline

```
Raw Social Media Text
        │
        ▼
[Stage 1: Dataset Ingestion & Validation]
        │
        ▼
[Stage 2 & 3: Deduplication & Binary Target Mapping] (44,378 clean rows)
        │
        ▼
[Stage 9: Leak-Free 70/15/15 Stratified Partitioning]
  ├── Train (31,064) ──► Fit Vectorizer & Models
  ├── Val   (6,657)  ──► Model Selection & Hyperparameter Tuning
  └── Test  (6,657)  ──► UNTOUCHED until final report
        │
        ▼
[Stage 4: Conservative Text Normalisation] (Lowercasing, URL/Mention tokens, Hashtag retention)
        │
        ▼
[Stage 8: Sublinear TF-IDF Character N-grams (3,5)]
        │
        ▼
[Stage 10: Calibrated Logistic Regression]
        │
        ▼
[Stage 6: Final Test Set Evaluation & Error Analysis]
        │
        ▼
[Inference: CivilityLens Moderation Interface] (Live Text Detection & Risk Categorisation)
```

---

## 7. Preprocessing Decisions

Social media text requires careful domain-specific normalisation:
1. **URL Normalisation:** Replaced with `urltoken` to eliminate random domain noise while signalling link inclusion.
2. **User Mention Normalisation:** Replaced with `usertoken` to prevent overfitting to specific target usernames.
3. **Hashtag Preservation:** Strips `#` while retaining the text (e.g. `#cyberbully` $\to$ `cyberbully`).
4. **Controlled Negation-Aware Stopwords:** Standard NLP libraries drop `not`, `no`, `never`. CivilityLens explicitly preserves negations to prevent inverting sentence polarity.
5. **Preserving Emotional Punctuation:** Repeated question/exclamation marks (`?!?!`) and emojis are preserved in the conservative baseline.

---

## 8. Feature Extraction: Bag of Words vs. TF-IDF

- **Bag of Words (CountVectorizer):** Encodes raw occurrences of terms. Sensitive to document length and dominated by common non-informative words.
- **TF-IDF (TfidfVectorizer):** Multiplies term frequency by inverse document frequency to downweight universal tokens and amplify distinctive terms.
- **Character N-grams ($3 \le n \le 5$):** Operates on character sequences across word boundaries (`char_wb`). Captures sub-word patterns, making it highly robust against obfuscated slurs (`l0ser`, `b!tch`).

---

## 9. Machine Learning Classifiers

1. **Multinomial Naive Bayes:** Probabilistic classifier applying Bayes' theorem with independence assumptions. Extremely fast, lightweight baseline.
2. **Linear Support Vector Machine (LinearSVC):** Convex optimization finding maximum-margin separating hyperplane. Highly effective in sparse text feature spaces.
3. **Logistic Regression:** Linear model with sigmoid link function. Provides calibrated continuous probabilities ($P(\text{cyberbullying} \mid \mathbf{x})$) essential for moderation risk triage.

---

## 10. Train, Validation, and Test Methodology

To eliminate **data snooping bias** and **data leakage**:
1. All exact duplicate rows and conflicting label pairs were purged *before* splitting.
2. Split ratio: **70% Training ($31,064$), 15% Validation ($6,657$), 15% Test ($6,657$)**.
3. Vectorizers were fitted **strictly on the Training split**.
4. Nine candidate pipelines were compared **strictly on the Validation split**.
5. The winning pipeline was evaluated **once on the untouched Test split**.

---

## 11. Measured Experimental Results

All experiments were executed on real data (`src/train.py`):

### Validation Comparison (Empirical Data):
| Experiment ID | Classifier | Feature Representation | Val Accuracy | Val Precision | Val Recall | Val F1-Score |
|---|---|---|---|---|---|---|
| **EXP-1A** | Multinomial NB | TF-IDF (1,2) | 0.8807 | 0.8879 | 0.9860 | 0.9344 |
| **EXP-1B** | Logistic Regression | TF-IDF (1,2) | 0.8951 | 0.9030 | 0.9830 | 0.9413 |
| **EXP-1C** | Linear SVM | TF-IDF (1,2) | 0.9000 | 0.9135 | 0.9740 | 0.9428 |
| **EXP-2A** | Multinomial NB | Bag of Words (1,2) | 0.8556 | 0.8997 | 0.9283 | 0.9138 |
| **EXP-2B** | Logistic Regression | Bag of Words (1,2) | 0.9004 | 0.9168 | 0.9708 | 0.9430 |
| **EXP-2C** | Linear SVM | Bag of Words (1,2) | 0.8771 | 0.9172 | 0.9400 | 0.9284 |
| **EXP-3** | Logistic Regression | TF-IDF (Unigram only) | 0.8976 | 0.9048 | 0.9831 | 0.9423 |
| **EXP-4** | Logistic Regression | TF-IDF (1,2) + Stopwords | 0.8936 | 0.9033 | 0.9807 | 0.9403 |
| **EXP-5** | **Logistic Regression** | **TF-IDF Char N-grams (3,5)** | **0.9028** | **0.9173** | **0.9742** | **0.9449** 🏆 |

### Final Untouched Test Set Evaluation (EXP-5):
- **Accuracy:** `89.88%`
- **Precision (Class 1):** `91.30%`
- **Recall (Class 1):** `97.45%`
- **F1-Score (Class 1):** `0.9427`
- **Macro F1:** `0.7531`

### Confusion Matrix (6,657 Test Records):
```
                       Predicted Non-Bullying (0)   Predicted Bullying (1)
Actual Non-Bullying (0)          435 (TN)                    529 (FP)
Actual Bullying (1)              145 (FN)                  5,548 (TP)
```

---

## 12. Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- macOS, Linux, or Windows

### Step 1: Clone or Open Project Directory
```bash
cd "/Users/anujshembekar/Library/Mobile Documents/com~apple~CloudDocs/Documents/Anuj/SEVENTH/NLP/LAB_PROJECT"
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 13. Training the Pipeline

To re-run data ingestion, cleaning, experimental validation, and test evaluation:
```bash
PYTHONPATH=. python3 src/train.py
```
This automatically updates:
- `artifacts/best_model_pipeline.joblib` (Fitted Pipeline artifact)
- `artifacts/evaluation_metrics.json` (Full test and validation metrics)
- `artifacts/experiment_comparison.csv` (Experiment leaderboard)
- `artifacts/error_analysis_samples.json` (Extracted test misclassifications)
- `artifacts/dataset_metadata.json` (Dataset dimensions and clean counts)

---

## 14. Launching the User Interface

CivilityLens includes two working UI options:

### Option A: Streamlit Web Application (Primary)
```bash
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`.

### Option B: Standalone Web Demo (Zero External Dependencies)
If Streamlit is not yet installed or you want a lightweight native demo:
```bash
PYTHONPATH=. python3 web_demo.py
```
Opens in your browser at `http://localhost:8501` using standard Python networking.

---

## 15. Project Directory Structure

```
CivilityLens/
├── app.py                      # Primary Streamlit Moderation Application
├── web_demo.py                 # Standalone native web demonstration server
├── requirements.txt            # Package dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Version control ignore rules
├── data/
│   ├── README.md               # Dataset provenance and schema details
│   └── cyberbullying_tweets.csv # Kaggle tweet dataset (47,692 rows)
├── src/
│   ├── __init__.py             # Package marker
│   ├── config.py               # Paths, constants, random seeds, label mappings
│   ├── data_loader.py          # Data ingestion, schema validation, cleaning
│   ├── preprocessing.py        # Conservative normaliser, stemming, stopwords
│   ├── train.py                # Model training, experiments, artifact saving
│   ├── evaluate.py             # Metrics calculation and confusion matrix
│   └── predict.py              # Inference engine, probability calibration, risk triage
├── artifacts/
│   ├── best_model_pipeline.joblib   # Trained pipeline artifact
│   ├── dataset_metadata.json        # Dataset statistics
│   ├── evaluation_metrics.json      # Full evaluation report
│   ├── experiment_comparison.csv    # Real experimental comparison table
│   └── error_analysis_samples.json  # False Positives & Negatives from test set
├── reports/
│   └── technical_report.md     # Academic research paper & methodology report
└── tests/
    ├── test_preprocessing.py   # Unit tests for text cleaning and tokenizers
    └── test_prediction.py      # Unit tests for pipeline inference and metrics
```

---

## 16. Running Unit Tests

Run the test suite using Python's standard `unittest` or `pytest`:
```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```
All 13 tests verify:
- Safe handling of empty, whitespace, and non-string inputs.
- URL and mention token substitution.
- Hashtag text preservation.
- Retention of negation words in custom stopwords.
- Model pipeline loading from disk and inference output structure.
- Graceful error handling for missing dataset paths.
- Accurate metric calculations and zero-division handling.

---

## 17. System Limitations

1. **Sarcasm and Indirect Hostility:** Statistical models evaluate token distributions and frequently miss polite irony (e.g. *"Great job, you really showed your incredible intelligence there"*).
2. **Context Window Absence:** Isolated tweets lack access to preceding thread interactions, making it difficult to distinguish offensive attacks from defensive counter-speech.
3. **Domain & Demographic Shift:** Models trained on 2020 Twitter data do not immediately generalize to novel Discord or gaming chat conventions without fine-tuning.

---

## 18. Future Improvements

- **Hierarchical Multiclass Routing:** A two-stage classifier that first screens for bullying, then categorizes the specific demographic target (`religion`, `gender`, `ethnicity`).
- **Conversational Context Tracking:** Incorporating preceding parent tweets to disambiguate quotes and counter-speech.
- **Explainable AI (XAI):** Integrating SHAP or LIME token highlighting directly into the Streamlit interface.

---

*CivilityLens — Natural Language Processing & Machine Learning Project.*
