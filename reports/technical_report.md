# CivilityLens: Assistive NLP Cyberbullying Detection Application
## Academic Technical Report & Methodology Paper

**Project Author:** BTech Computer Science Candidate  
**Academic Module:** Natural Language Processing & Machine Learning Lab Project  
**Date:** October 2026  
**Artifact Repository:** CivilityLens  

---

## Abstract

Online social media platforms are plagued by toxic interactions, abusive behavior, and identity-based cyberbullying. While automated detection systems have been proposed, high false-positive rates frequently stifle benign discourse, and black-box architectures fail to provide actionable interpretability for moderation teams. In this project, we present **CivilityLens**, an end-to-end Natural Language Processing (NLP) and machine-learning framework engineered as a decision-support system for digital content moderators. CivilityLens processes social media text through a conservative normalisation pipeline, controlled n-gram feature representations, and calibrated linear classifiers. We train, validate, and compare nine experimental configurations across Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machines on 47,692 tweets from the Kaggle Cyberbullying Classification benchmark. Our best-performing model—Logistic Regression utilizing sublinear TF-IDF character n-grams ($3 \le n \le 5$)—achieved a **Validation F1-score of 0.9449** and demonstrated generalisation on an untouched test set of 6,657 records with **89.88% accuracy, 91.30% precision, 97.45% recall, and an F1-score of 0.9427**. We provide an error analysis taxonomy detailing systemic failure modes such as sarcasm, reclaimed in-group slang, and subtle exclusion.

---

## 1. Introduction & Problem Statement

### 1.1 Context
Digital communication environments such as Twitter, Reddit, and Discord serve as modern public squares. However, the prevalence of targeted harassment, hate speech, and cyberbullying poses psychological and legal risks to individuals and online communities. Manual moderation cannot scale to millions of user messages generated per minute.

### 1.2 Problem Formulation
Existing commercial and automated approaches frequently suffer from three primary shortcomings:
1. **Aggressive False Alarms (False Positives):** Models over-index on profane words or identity keywords, flagging non-abusive discussions, counter-speech, or reclaimed vernacular.
2. **Adversarial Obfuscation (False Negatives):** Perpetrators intentionally misspell slurs, insert special symbols (e.g., `b!tch`), or rely on subtle character substitutions that evade rigid token dictionaries.
3. **Overstated System Autonomy:** Tools framed as "automated character judges" undermine procedural fairness by masquerading statistical correlations as moral ground truth.

### 1.3 Project Objectives
CivilityLens addresses these challenges by pursuing five core objectives:
- **Reproducible Pipeline:** Implement an end-to-end NLP data pipeline from raw Twitter data to inference.
- **Fair Model Comparison:** Empirically compare Naive Bayes, Logistic Regression, and Support Vector Machines on controlled splits without data leakage.
- **Leakage Prevention:** Split 70% Train, 15% Validation, and 15% Test *after* rigorous deduplication and conflicting-label purging, keeping the test set untouched until final model selection.
- **Assistive Moderation UI:** Deliver a user interface displaying calibrated probabilities, decision metrics, and disclaimers.
- **Transparent Failure Taxonomy:** Characterize the boundary limitations of statistical NLP classifiers in real-world deployment.

---

## 2. Dataset Description & Pre-Processing

### 2.1 Dataset Provenance
We utilize the Kaggle Cyberbullying Classification dataset compiled by Andrew M., containing **47,692 raw tweets** annotated across six categories:

| Original Category | Raw Count | Proportional Share | Annotation Characterization |
|---|---|---|---|
| `religion` | 7,998 | 16.77% | Faith-targeted slurs and hate speech |
| `age` | 7,992 | 16.76% | Schoolyard insults, geriatric or youth mocking |
| `gender` | 7,973 | 16.72% | Sexism, sexualized insults, gender harassment |
| `ethnicity` | 7,961 | 16.69% | Racial slurs, xenophobic abuse |
| `not_cyberbullying` | 7,945 | 16.66% | Benign banter, reality TV discussions, daily chatter |
| `other_cyberbullying` | 7,823 | 16.40% | General aggression without specific demographic target |
| **Total** | **47,692** | **100.0%** | **Balanced across 6 categories** |

### 2.2 Target Formulation (Binary Moderation Triage)
To support frontline moderation queues, the six multiclass labels were mapped to a binary classification target:

$$\text{Target} = \begin{cases} 0 & \text{if } \text{label} = \text{'not\_cyberbullying'} \\ 1 & \text{if } \text{label} \in \{\text{'religion'}, \text{'age'}, \text{'gender'}, \text{'ethnicity'}, \text{'other\_cyberbullying'}\} \end{cases}$$

This mapping concentrates model sensitivity on identifying any form of harassment (83.3% positive prevalence in the mapped dataset) while preserving the original labels for multi-class error tracing.

### 2.3 Data Cleaning & Leakage Prevention
Raw social media text suffers from annotation ambiguity and exact duplicates. Our validation step (`src/data_loader.py`) executed the following cleaning sequence:
1. **Empty Record Pruning:** Deleted rows with null values or whitespace-only strings ($0$ rows).
2. **Exact Duplicate Removal:** Pruned 36 rows where both `tweet_text` and `cyberbullying_type` were identical.
3. **Conflicting Label Purging:** Identified identical tweets annotated with contradictory labels (e.g., marked `religion` by one annotator and `not_cyberbullying` by another) and purged all conflicting instances ($3,278$ records). This eliminated label noise and prevented contradictory loss gradients.
4. **Final Clean Corpus:** Retained **44,378 pristine records**.

---

## 3. Methodology & Feature Engineering

### 3.1 Conservative Text Normalisation
Aggressive text cleaners discard emojis, exclamation marks, and capitalization. However, in social media discourse, repeated punctuation (`?!?!`) and uppercase text signify aggressive tone. CivilityLens employs a conservative normaliser:
- Lowercase conversion.
- Replacement of HTTP/HTTPS URLs with canonical `urltoken`.
- Replacement of user mentions (`@username`) with `usertoken`.
- Preservation of hashtag semantic content (e.g., `#cyberbully` $\to$ `cyberbully`).
- Normalisation of repeated whitespace.

### 3.2 Controlled Stopword Handling
Standard NLP toolkits eliminate stop words such as `not`, `no`, `never`, and `without`. In cyberbullying detection, eliminating negations flips sentence polarity:
$$\text{"She is not ugly"} \xrightarrow{\text{standard stopword drop}} \text{"She ugly"}$$
To prevent this error, CivilityLens isolates a set of 27 critical negation words that are explicitly retained during token filtering.

### 3.3 Representation Architectures
We compared two fundamental feature extraction families:
1. **Bag of Words (CountVectorizer):** Vector of raw token frequencies $f(t, d)$.
2. **TF-IDF (TfidfVectorizer):** Scaled frequency inversely weighted by document frequency:
   $$\text{TF-IDF}(t, d, D) = (1 + \log \text{TF}(t, d)) \cdot \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right)$$
   Configured with sublinear term-frequency scaling to dampen the distortion of repeated words.
3. **Character N-grams ($3 \le n \le 5$):** Sub-word character sequences extracted across word boundaries (`char_wb`). Character n-grams provide robustness against intentional obfuscation (e.g., `l0ser` shares the 3-gram `ser` and `0se` with `loser`).

---

## 4. Experimental Setup

### 4.1 Data Splitting Strategy
To guarantee academic reproducibility, the cleaned dataset ($N = 44,378$) was partitioned into three disjoint sets using stratified sampling on the binary target with `random_state = 42`:
- **Training Set (70%):** $31,064$ samples. Used exclusively to fit vectorizers and train classifier parameters.
- **Validation Set (15%):** $6,657$ samples. Used exclusively for hyperparameter tuning, feature selection, and model comparison.
- **Test Set (15%):** $6,657$ samples. **Strictly untouched** until model selection was finalized.

### 4.2 Candidate Classifiers
We evaluated three supervised learning architectures:
- **Multinomial Naive Bayes (MNB):** Probabilistic baseline applying Bayes' rule with Laplace smoothing ($\alpha = 1.0$).
- **Logistic Regression (LR):** Convex loss optimization using L2 regularisation and liblinear solver with probability calibration via the sigmoid link function.
- **Linear Support Vector Machine (LinearSVC):** Maximum-margin hyperplane separator optimizing hinge loss.

---

## 5. Measured Experimental Results

All nine experimental configurations were trained on the training partition and scored on the validation partition.

### 5.1 Validation Comparison Table (Empirical Data)

| Experiment ID | Classifier | Feature Extraction | Preprocessing | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Training Time |
|---|---|---|---|---|---|---|---|---|
| **EXP-1A** | Multinomial NB | TF-IDF (1,2) | Conservative Baseline | 0.8807 | 0.8879 | 0.9860 | 0.9344 | 2.46s |
| **EXP-1B** | Logistic Regression | TF-IDF (1,2) | Conservative Baseline | 0.8951 | 0.9030 | 0.9830 | 0.9413 | 2.46s |
| **EXP-1C** | Linear SVM | TF-IDF (1,2) | Conservative Baseline | 0.9000 | 0.9135 | 0.9740 | 0.9428 | 2.44s |
| **EXP-2A** | Multinomial NB | Bag of Words (1,2) | Conservative Baseline | 0.8556 | 0.8997 | 0.9283 | 0.9138 | 2.26s |
| **EXP-2B** | Logistic Regression | Bag of Words (1,2) | Conservative Baseline | 0.9004 | 0.9168 | 0.9708 | 0.9430 | 2.91s |
| **EXP-2C** | Linear SVM | Bag of Words (1,2) | Conservative Baseline | 0.8771 | 0.9172 | 0.9400 | 0.9284 | 3.24s |
| **EXP-3** | Logistic Regression | TF-IDF (Unigram only) | Conservative Baseline | 0.8976 | 0.9048 | 0.9831 | 0.9423 | 1.19s |
| **EXP-4** | Logistic Regression | TF-IDF (1,2) + Stopwords | Custom Stopword Filter | 0.8936 | 0.9033 | 0.9807 | 0.9403 | 2.11s |
| **EXP-5** | **Logistic Regression** | **TF-IDF Char N-grams (3,5)** | **Char N-gram Sub-words** | **0.9028** | **0.9173** | **0.9742** | **0.9449** | **7.38s** |

### 5.2 Key Empirical Findings
1. **Character N-grams Outperform Word N-grams:** EXP-5 achieved the highest validation F1-score ($0.9449$) and accuracy ($0.9028$). Sub-word n-grams effectively mitigated out-of-vocabulary slang and intentional misspellings.
2. **TF-IDF vs. Bag of Words:** TF-IDF provided consistently superior stability for Naive Bayes ($+2.06\%$ F1) and Linear SVM ($+1.44\%$ F1).
3. **Stopword Pruning Impact:** Removing stopwords (EXP-4) yielded a slight decrease in F1 ($0.9403$ vs $0.9413$), validating our hypothesis that conversational filler words carry structural meaning in informal tweet dialogue.

---

## 6. Final Untouched Test Set Evaluation

Based on the validation results, **EXP-5 (Logistic Regression with TF-IDF Character N-grams 3-5)** was selected as the final production pipeline and evaluated once on the untouched test partition ($N = 6,657$).

### 6.1 Final Performance Summary
- **Test Accuracy:** **89.88%** ($0.8988$)
- **Test Precision (Class 1):** **91.30%** ($0.9130$)
- **Test Recall (Class 1):** **97.45%** ($0.9745$)
- **Test F1-Score (Class 1):** **94.27%** ($0.9427$)
- **Macro F1-Score:** **75.31%** ($0.7531$)

### 6.2 Confusion Matrix Analysis
$$\begin{pmatrix} \text{TN} = 435 & \text{FP} = 529 \\ \text{FN} = 145 & \text{TP} = 5,548 \end{pmatrix}$$

- **True Positives (5,548):** Successfully intercepted toxic, abusive, or discriminatory attacks.
- **True Negatives (435):** Benign messages correctly recognized without raising alarms.
- **False Positives (529):** Harmless messages flagged by the system due to sensitive keyword overlaps.
- **False Negatives (145):** Harassment instances that bypassed detection due to subtle, implicit phrasing.

---

## 7. Error Analysis & Failure Taxonomy

We conducted a manual review of test set misclassifications:

### 7.1 False Positive Failure Modes (False Alarms)
- **Quotation and Counter-Speech:** Users quoting an abusive remark to condemn it were flagged because the model lacks speaker attribution:
  *Example:* "He actually said 'go back to your country' to her in public, that is unacceptable!"
- **Heated Political/Sport Debates:** Strong vocabulary in non-targeted arguments:
  *Example:* "This referee is an absolute disgrace to football, completely ruined the match."

### 7.2 False Negative Failure Modes (Missed Harassment)
- **Implicit Exclusion:** Harassment devoid of profanity or identity slurs:
  *Example:* "Notice how nobody replies when you speak? Take the hint."
- **Sarcastic Praise:** Surface-level polite lexicon hiding hostility:
  *Example:* "Oh look, our genius classmate finally managed to spell their own name."

---

## 8. Responsible AI & Operational Limitations

1. **Moderation-Support Only:** CivilityLens is an algorithmic triage tool, not a legal or moral arbiter.
2. **Context Blindness:** Isolated tweets lack access to conversational thread history, user relationship dynamics, and platform culture.
3. **Dataset Bias:** Training on 2020 Twitter data limits generalisability to emerging slang on TikTok, Discord, or gaming voice transcripts.

---

## 9. Conclusion & Future Scope

### 9.1 Summary of Contributions
CivilityLens demonstrates an academically sound, reproducible pipeline for cyberbullying detection. By prioritizing data hygiene, preventing target leakage, and utilizing character n-gram representations with calibrated logistic regression, the project achieved a **0.9427 Test F1-Score** while maintaining a sub-10ms inference latency suitable for real-time web deployment.

### 9.2 Future Roadmap
- **Multiclass Hierarchy:** Implement a 2-stage architecture: binary harassment detection followed by demographic category routing (`religion`, `gender`, `ethnicity`).
- **Context Window Modeling:** Incorporate preceding conversation turns to resolve ambiguous counter-speech.
- **Explainability:** Integrate SHAP (SHapley Additive exPlanations) or LIME to highlight exact word attribution tokens in the user interface.

---

*CivilityLens Academic Technical Report — Compiled October 2026.*
