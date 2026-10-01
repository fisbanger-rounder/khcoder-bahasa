"""Pure Bahasa Indonesia text helpers for IndoTextMiner.

Deliberately free of Streamlit imports so the logic stays unit-testable.
"""
import re
from functools import lru_cache

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

CUSTOM_STOPWORDS = {'ya', 'tidak', 'ada', 'dan', 'yang', 'di', 'ke', 'dari', 'ini', 'itu'}


@lru_cache(maxsize=1)
def get_stopwords():
    return set(StopWordRemoverFactory().get_stop_words()) | CUSTOM_STOPWORDS


@lru_cache(maxsize=1)
def get_stemmer():
    return StemmerFactory().create_stemmer()


def preprocess_text(text, do_stemming=False):
    """Lowercase, strip punctuation, drop stopwords and 1-2 character tokens.

    Digits are kept: years, prices and percentages carry meaning in text mining.
    """
    if not isinstance(text, str):
        return []
    tokens = re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()
    stopwords = get_stopwords()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    if do_stemming:
        stemmer = get_stemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def get_kwic(text_series, keyword, window=5):
    """Keyword in Context (KWIC): one row per occurrence of `keyword`."""
    pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
    kwic_results = []

    for idx, text in enumerate(text_series):
        if not isinstance(text, str):
            continue
        words = text.split()
        for i, word in enumerate(words):
            if pattern.search(word):
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                kwic_results.append({
                    "Doc ID": idx + 1,
                    "Konteks Kiri (Left)": " ".join(words[start:i]),
                    "Kata Kunci (Keyword)": words[i],
                    "Konteks Kanan (Right)": " ".join(words[i + 1:end]),
                })
    return pd.DataFrame(kwic_results)
