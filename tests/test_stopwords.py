import pytest

from stopwords import ENGLISH_STOPWORDS, load_stopwords


def test_english_list_covers_function_words():
    assert {"the", "of", "a", "and", "is"} <= ENGLISH_STOPWORDS


def test_english_list_keeps_content_words():
    # Some popular lists drop these; they can be real vocabulary in a word-association game.
    assert not ({"table", "fire", "bill", "top", "bank"} & ENGLISH_STOPWORDS)


def test_english_list_covers_contraction_fragments():
    # The tokenizer splits on apostrophes, so "don't" arrives as "don" and "t".
    assert {"don", "t", "s", "ll", "ve"} <= ENGLISH_STOPWORDS


def test_load_english_and_none():
    assert load_stopwords("english") is ENGLISH_STOPWORDS
    assert load_stopwords("none") == frozenset()


def test_load_from_file_ignores_comments_blanks_and_case(tmp_path):
    path = tmp_path / "custom.txt"
    path.write_text("# my list\nFoo\n\n  bar  \n", encoding="utf-8")

    assert load_stopwords(str(path)) == frozenset({"foo", "bar"})


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_stopwords(str(tmp_path / "does_not_exist.txt"))
