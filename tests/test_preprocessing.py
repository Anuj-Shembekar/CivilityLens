"""Unit tests for text preprocessing and normalization in CivilityLens."""

import unittest
from src.preprocessing import (
    clean_text_conservative,
    apply_stemming,
    apply_lemmatization,
    get_custom_stopwords,
    URL_PLACEHOLDER,
    MENTION_PLACEHOLDER,
)


class TestPreprocessing(unittest.TestCase):

    def test_clean_text_empty_and_non_string(self):
        """Verify empty and non-string inputs are safely handled."""
        self.assertEqual(clean_text_conservative(""), "")
        self.assertEqual(clean_text_conservative("   "), "")
        self.assertEqual(clean_text_conservative(None), "")
        self.assertEqual(clean_text_conservative(12345), "12345")

    def test_url_replacement(self):
        """Verify standard HTTP, HTTPS, and WWW URLs are replaced by token."""
        text1 = "Check this out https://example.com/page now!"
        self.assertIn(URL_PLACEHOLDER, clean_text_conservative(text1))
        self.assertNotIn("https://example.com/page", clean_text_conservative(text1))

        text2 = "Visit www.example.org for more info"
        self.assertIn(URL_PLACEHOLDER, clean_text_conservative(text2))

    def test_user_mention_replacement(self):
        """Verify @mentions are replaced by token."""
        text = "@john_doe and @JaneDoe are talking"
        cleaned = clean_text_conservative(text)
        self.assertIn(MENTION_PLACEHOLDER, cleaned)
        self.assertNotIn("@john_doe", cleaned)
        self.assertNotIn("@JaneDoe", cleaned)

    def test_hashtag_preservation(self):
        """Verify # symbol is stripped while preserving the hashtag text."""
        text = "Stop this #cyberbullying and #harassment right now"
        cleaned = clean_text_conservative(text)
        self.assertNotIn("#", cleaned)
        self.assertIn("cyberbullying", cleaned)
        self.assertIn("harassment", cleaned)

    def test_whitespace_normalization(self):
        """Verify repeated spaces, newlines, and tabs are collapsed to a single space."""
        text = "Hello \n\n   world \t  test   again! "
        cleaned = clean_text_conservative(text)
        self.assertEqual(cleaned, "hello world test again!")

    def test_custom_stopwords_retain_negations(self):
        """Verify crucial negation terms are retained in the custom stopwords list."""
        sw = get_custom_stopwords()
        self.assertNotIn("not", sw)
        self.assertNotIn("no", sw)
        self.assertNotIn("never", sw)
        self.assertNotIn("without", sw)
        self.assertIn("the", sw)
        self.assertIn("is", sw)
        self.assertIn("at", sw)

    def test_stemming_and_lemmatization(self):
        """Verify stemming and lemmatization functions produce expected outputs."""
        sample = "The bullies are bullying innocent students"
        stemmed = apply_stemming(sample)
        lemmatized = apply_lemmatization(sample)

        self.assertIn("bulli", stemmed)
        self.assertGreater(len(lemmatized), 0)


if __name__ == "__main__":
    unittest.main()
