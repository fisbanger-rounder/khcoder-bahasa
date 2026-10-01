"""Self-checks for the pure text logic. Run: python test_textminer.py"""
import pandas as pd

from textminer import get_kwic, preprocess_text


def test_non_string_is_empty():
    assert preprocess_text(None) == []
    assert preprocess_text(42) == []


def test_punctuation_stripped_digits_kept():
    tokens = preprocess_text("Panen 2024: hasilnya 60%!")
    assert "2024" in tokens
    assert "60" in tokens
    assert all(t.isalnum() for t in tokens)


def test_stopwords_and_short_tokens_dropped():
    assert preprocess_text("yang di ke ini itu ada ya tidak") == []
    assert preprocess_text("di ke") == []


def test_stemming_is_one_to_one():
    plain = preprocess_text("petani memanen tanaman melon")
    stemmed = preprocess_text("petani memanen tanaman melon", do_stemming=True)
    assert plain, "expected non-stopword content tokens"
    # stem() maps token -> token, so it must never change the token count
    assert len(stemmed) == len(plain)


def test_kwic_window_and_bounds():
    df = get_kwic(pd.Series(["satu dua tiga empat lima enam"]), "empat", window=2)
    assert len(df) == 1
    assert df.iloc[0]["Konteks Kiri (Left)"] == "dua tiga"
    assert df.iloc[0]["Konteks Kanan (Right)"] == "lima enam"


def test_kwic_at_start_does_not_underflow():
    df = get_kwic(pd.Series(["satu dua tiga"]), "satu", window=5)
    assert df.iloc[0]["Konteks Kiri (Left)"] == ""
    assert df.iloc[0]["Konteks Kanan (Right)"] == "dua tiga"


def test_kwic_case_insensitive_missing_and_non_string():
    assert len(get_kwic(pd.Series(["Budi dan budi"]), "BUDI")) == 2
    assert get_kwic(pd.Series(["tidak ada"]), "zzz").empty
    assert get_kwic(pd.Series([None, 123]), "budi").empty


if __name__ == "__main__":
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} passed")
