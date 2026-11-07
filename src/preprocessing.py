"""Text Preprocessing and Normalization Pipeline for CivilityLens.

Implements:
- Stage 4: Conservative Text Normalisation (Baseline)
- Stage 6: Controlled Negation-Aware Stopword Removal (Optional Experiment)
- Stage 7: Stemming and Lemmatisation (Optional Experiments)

Designed to be reusable across training, validation, testing, and live inference.
"""

import re
from pathlib import Path
from typing import Callable, List, Optional
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Gracefully handle optional NLTK dependency
try:
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    # Register local workspace nltk_data path if present
    LOCAL_NLTK_DATA = Path(__file__).resolve().parent.parent / "nltk_data"
    if LOCAL_NLTK_DATA.exists() and str(LOCAL_NLTK_DATA) not in nltk.data.path:
        nltk.data.path.insert(0, str(LOCAL_NLTK_DATA))
    NLTK_AVAILABLE = True
except ImportError:
    nltk = None
    PorterStemmer = None
    WordNetLemmatizer = None
    NLTK_AVAILABLE = False

# Regex patterns for normalization
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_REGEX = re.compile(r"@\w+", re.IGNORECASE)
HASHTAG_REGEX = re.compile(r"#(\w+)")
WHITESPACE_REGEX = re.compile(r"\s+")
HTML_TAG_REGEX = re.compile(r"<.*?>")

# Placeholders
URL_PLACEHOLDER = "urltoken"
MENTION_PLACEHOLDER = "usertoken"

# Safe Stopword Set (Preserves negations and relational pronouns that carry bullying intent)
NEGATION_WORDS = {
    "not", "no", "nor", "neither", "never", "none", "nobody", "nowhere",
    "nothing", "hardly", "scarcely", "barely", "doesn't", "don't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
    "won't", "wouldn't", "cannot", "couldn't", "shouldn't", "without"
}


def clean_text_conservative(text: str) -> str:
    """Stage 4: Conservative text normalization baseline.

    Steps:
    1. Safe non-string handling.
    2. Lowercase conversion.
    3. Remove HTML artifacts.
    4. Replace URLs with URL_PLACEHOLDER.
    5. Replace @mentions with MENTION_PLACEHOLDER.
    6. Strip '#' from hashtags while retaining the word itself.
    7. Normalize repeated whitespace.

    Note: Emojis and punctuation are deliberately preserved in the conservative
    baseline, as punctuation marks (e.g. repeated '?!') and emojis can convey
    emotional aggression or distress in social media discourse.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Lowercase
    cleaned = text.lower()

    # Remove HTML remnants
    cleaned = HTML_TAG_REGEX.sub(" ", cleaned)

    # Replace URLs
    cleaned = URL_REGEX.sub(f" {URL_PLACEHOLDER} ", cleaned)

    # Replace User Mentions
    cleaned = MENTION_REGEX.sub(f" {MENTION_PLACEHOLDER} ", cleaned)

    # Preserve hashtag words (e.g., #loser -> loser)
    cleaned = HASHTAG_REGEX.sub(r" \1 ", cleaned)

    # Normalize whitespace
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()

    return cleaned


def get_custom_stopwords() -> List[str]:
    """Return English stopwords while retaining critical negation words."""
    try:
        from nltk.corpus import stopwords
        words_set = set(stopwords.words("english"))
    except Exception:
        words_set = set(ENGLISH_STOP_WORDS)

    # Remove negations from stopwords so they are NOT deleted
    safe_stopwords = sorted(list(words_set - NEGATION_WORDS))
    return safe_stopwords


def apply_stemming(text: str) -> str:
    """Stage 7: Porter Stemmer experiment.

    Reduces words to crude morphological stems using algorithmic suffix rules.
    Example: 'bullying', 'bullied' -> 'bulli'
    """
    if PorterStemmer is None:
        return f"[NLTK not installed in active environment. Run: pip install nltk]"
    stemmer = PorterStemmer()
    cleaned = clean_text_conservative(text)
    tokens = cleaned.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


def apply_lemmatization(text: str) -> str:
    """Stage 7: WordNet Lemmatization experiment.

    Reduces words to canonical dictionary lemmas using lexical databases.
    Default WordNet lookup treats tokens as nouns unless POS tagged.
    """
    if WordNetLemmatizer is None:
        return f"[NLTK not installed in active environment. Run: pip install nltk]"
    try:
        lemmatizer = WordNetLemmatizer()
        # Verify WordNet is accessible
        lemmatizer.lemmatize("testing")
    except Exception:
        try:
            if nltk:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            lemmatizer = WordNetLemmatizer()
        except Exception:
            return clean_text_conservative(text)

    cleaned = clean_text_conservative(text)
    tokens = cleaned.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)


def get_preprocessor_by_name(strategy_name: str) -> Callable[[str], str]:
    """Retrieve preprocessor function by experimental strategy name."""
    strategies = {
        "conservative": clean_text_conservative,
        "stemming": apply_stemming,
        "lemmatization": apply_lemmatization,
    }
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown preprocessing strategy: '{strategy_name}'. "
            f"Available options: {list(strategies.keys())}"
        )
    return strategies[strategy_name]


if __name__ == "__main__":
    sample = (
        "RT @victim: Why do you hate me so much?! Check out https://t.co/evil "
        "and tell me I am #not_ugly and never deserved this!! >:("
    )
    print("Original:     ", sample)
    print("Conservative: ", clean_text_conservative(sample))
    print("Stemmed:      ", apply_stemming(sample))
    print("Lemmatized:   ", apply_lemmatization(sample))
