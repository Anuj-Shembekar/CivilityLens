# CivilityLens Dataset Guide

This directory hosts the cyberbullying tweet corpus used to train and evaluate the CivilityLens NLP pipeline.

---

## 1. Dataset Provenance & Attribution

- **Dataset Name:** Cyberbullying Classification
- **Curator:** Andrew M. (`andrewmvd`)
- **Source:** [Kaggle Dataset Repository](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)
- **Primary File:** `cyberbullying_tweets.csv`
- **Total Records (Raw):** 47,692
- **License:** CC BY 4.0 International

---

## 2. Schema Specification

The dataset contains two primary columns:

| Column Name | Data Type | Description |
|---|---|---|
| `tweet_text` | `object` (string) | Raw text scraped from public Twitter interactions. |
| `cyberbullying_type` | `object` (string) | Original categorical label assigned during annotation. |

---

## 3. Original Categorical Distribution

Across the 47,692 raw rows, the original class distribution is balanced as follows:

| Class Label | Count | Proportion | Nature of Content |
|---|---|---|---|
| `religion` | 7,998 | 16.77% | Harassment targeting religious faith and practices. |
| `age` | 7,992 | 16.76% | Bullying targeting youth, schoolyard context, or elderly age. |
| `gender` | 7,973 | 16.72% | Sexist, misogynistic, or gender-based harassment. |
| `ethnicity` | 7,961 | 16.69% | Racist slurs, xenophobic attacks, and ethnic harassment. |
| `not_cyberbullying` | 7,945 | 16.66% | Benign social conversation, television commentary, casual banter. |
| `other_cyberbullying` | 7,823 | 16.40% | General harassment not falling neatly into the 4 demographic buckets. |

---

## 4. Binary Target Mapping (Engineering Decision)

To establish an assistive moderation triage pipeline, the project projects the six classes into a binary decision:

$$\text{Target} = \begin{cases} 0 & \text{if } \text{cyberbullying\_type} = \text{'not\_cyberbullying'} \\ 1 & \text{if } \text{cyberbullying\_type} \in \{\text{'religion'}, \text{'age'}, \text{'gender'}, \text{'ethnicity'}, \text{'other\_cyberbullying'}\} \end{cases}$$

### Trade-offs:
- **Advantage:** Transforms the problem into an actionable binary classification task suitable for rapid frontline moderation screening.
- **Limitation:** Collapses demographic-specific harassment distinctions into a single positive class. Future iterations can implement hierarchical multi-class routing.

---

## 5. Cleaning & Deduplication Integrity

During data loading (`src/data_loader.py`):
1. **Empty strings / Whitespace-only text:** Purged.
2. **Exact duplicates:** 36 exact duplicate rows (text and label both identical) were dropped.
3. **Conflicting labels on identical text:** Identical tweets assigned conflicting category annotations were dropped to eliminate contradictory gradient updates.
4. **Leakage Prevention:** Deduplication occurs *prior* to the 70/15/15 stratified train/val/test split to eliminate train-test bleed.
5. **Retained clean samples:** 44,378 rows.
