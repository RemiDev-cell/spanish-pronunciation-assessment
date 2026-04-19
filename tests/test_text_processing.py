from src.text_processing import build_word_expectations, normalize_expected_text, tokenize_words


def test_normalize_expected_text():
    raw = "  Hola,   mundo   "
    assert normalize_expected_text(raw) == "Hola, mundo"


def test_tokenize_words():
    n = normalize_expected_text("El murciélago veloz.")
    toks = tokenize_words(n)
    assert "murciélago" in toks


def test_silabeador_expectations():
    n = normalize_expected_text("murciélago")
    ex = build_word_expectations(n)
    assert len(ex) == 1
    assert len(ex[0].syllables) >= 2
    assert 0 <= ex[0].stressed_syllable_index < len(ex[0].syllables)
