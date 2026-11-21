"""Tests for the text preprocessing in features.py and how the
vectoriser uses it — the cleaning rules, and the TF-IDF settings
that decide what the model actually sees.

These tests pin behaviour that the earlier test files do not
cover: exactly what ``clean_text`` keeps and drops, and that the
training pipeline applies the same cleaning (lower-casing,
whitespace collapse) with word pairs, no stopword removal, and
no minimum document frequency.
"""

import pytest

from ticketsense.features import clean_text
from ticketsense.predict import predict_label
from ticketsense.train import build_pipeline


def test_clean_text_collapses_tabs_newlines_and_mixed_whitespace():
    assert clean_text("charged\tTWICE\nfor  my\n\nplan") == "charged twice for my plan"
    assert clean_text("\r\n  hello   world \t ") == "hello world"


def test_clean_text_keeps_punctuation_and_numbers():
    # Cleaning is only lower-casing and whitespace — punctuation
    # and order numbers survive, so the model can still see them.
    assert clean_text("Charged, TWICE! Order #12345?") == "charged, twice! order #12345?"
    assert clean_text("My parcel 98765 has NOT arrived.") == "my parcel 98765 has not arrived."


def test_clean_text_keeps_non_ascii_text():
    # Real tickets mix languages; cleaning must not drop or mangle
    # non-ASCII words (Roman Urdu, accented names).
    assert clean_text("Mera ORDER nahi AAYA") == "mera order nahi aaya"
    assert clean_text("Café   PRØBLEM") == "café prøblem"


def test_clean_text_is_idempotent_and_keeps_a_single_word():
    assert clean_text("a") == "a"
    once = clean_text("  Charged   TWICE  ")
    assert clean_text(once) == once == "charged twice"


def test_clean_text_rejects_every_non_string_and_blank_form():
    for bad in (None, 123, 3.5, ["charged twice"], {"text": "charged twice"}, b"charged twice"):
        with pytest.raises(ValueError, match="must be a string"):
            clean_text(bad)  # type: ignore[arg-type]
    for blank in ("", "   ", "\n\t\r ", " \u00a0 "):
        with pytest.raises(ValueError, match="must not be empty"):
            clean_text(blank)


def test_clean_text_keeps_punctuation_only_text():
    # "!!!" is not blank — it passes cleaning and is the model's
    # problem (an all-zero TF-IDF row), not the cleaner's.
    assert clean_text("!!! ??? ...") == "!!! ??? ..."


def test_vectoriser_uses_clean_text_as_its_preprocessor():
    pipeline = build_pipeline()
    assert pipeline.named_steps["tfidf"].preprocessor is clean_text


def test_vectoriser_vocabulary_is_all_lowercase():
    pipeline = build_pipeline()
    pipeline.fit(
        ["Charged TWICE for my ORDER", "The APP Crashes Every Time"],
        ["billing", "technical"],
    )
    vocabulary = set(pipeline.named_steps["tfidf"].get_feature_names_out())
    assert "charged" in vocabulary
    assert "twice" in vocabulary
    assert "app" in vocabulary
    assert not any(word != word.lower() for word in vocabulary)


def test_vectoriser_includes_word_pairs():
    pipeline = build_pipeline()
    pipeline.fit(
        ["charged twice please refund", "app crashes every time"],
        ["billing", "technical"],
    )
    vocabulary = set(pipeline.named_steps["tfidf"].get_feature_names_out())
    assert "charged twice" in vocabulary
    assert "app crashes" in vocabulary


def test_vectoriser_keeps_rare_words_and_stopwords():
    # min_df=1 and no stopword list: a word seen once, and plain
    # words like "the", both stay in the vocabulary.
    pipeline = build_pipeline()
    pipeline.fit(
        ["charged twice for the zebra parcel", "app crashes for the user"],
        ["billing", "technical"],
    )
    vocabulary = set(pipeline.named_steps["tfidf"].get_feature_names_out())
    assert "zebra" in vocabulary
    assert "the" in vocabulary
    assert "for" in vocabulary


def test_vectoriser_tokenises_punctuation_away():
    # clean_text keeps the comma, but the TF-IDF tokeniser splits
    # on it, so "charged," and "charged" become the same feature.
    pipeline = build_pipeline()
    pipeline.fit(
        ["charged, twice! please refund.", "app crashes, every time!"],
        ["billing", "technical"],
    )
    vocabulary = set(pipeline.named_steps["tfidf"].get_feature_names_out())
    assert "charged" in vocabulary
    assert "charged," not in vocabulary
    assert "twice" in vocabulary


def test_predict_label_hands_the_pipeline_cleaned_text():
    """The pipeline must see the cleaned text, not the raw input."""

    class _RecordingPipeline:
        def __init__(self):
            self.seen = []

        def predict(self, texts):
            self.seen.extend(texts)
            return ["billing"] * len(texts)

        def predict_proba(self, texts):
            return [[0.8, 0.2]] * len(texts)

    pipeline = _RecordingPipeline()
    label, confidence = predict_label(pipeline, "  Charged   TWICE  ")
    assert label == "billing"
    assert confidence == 0.8
    assert pipeline.seen == ["charged twice"]
