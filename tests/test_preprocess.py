import json

import numpy as np
import pytest

import preprocess


def _write_corpus(path, text):
    path.write_text(text, encoding="utf-8")


def test_token_streamer_lowercases_and_strips_punctuation(tmp_path):
    file_path = tmp_path / "sample.txt"
    _write_corpus(file_path, "Hello, World! Hello again.")

    tokens = list(preprocess.token_streamer(file_path))

    assert tokens == ["hello", "world", "hello", "again"]


def test_corpus_streamer_reads_directory_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    _write_corpus(tmp_path / "a.txt", "one two")
    _write_corpus(tmp_path / "sub" / "b.txt", "three")

    tokens = sorted(preprocess.corpus_streamer(tmp_path))

    assert tokens == ["one", "three", "two"]


def test_corpus_streamer_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        list(preprocess.corpus_streamer("/definitely/not/a/real/path"))


def test_preprocess_corpus_filters_words_below_min_freq(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "MIN_FREQ", 2)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    _write_corpus(raw_dir / "corpus.txt", "cat cat dog cat")

    preprocess.preprocess_corpus(raw_dir=str(raw_dir), processed_dir=str(processed_dir))

    vocab = json.loads((processed_dir / "sorted_vocab.json").read_text(encoding="utf-8"))

    assert "cat" in vocab
    assert "dog" not in vocab  # only occurred once, below the patched MIN_FREQ
    cat_id, cat_count = vocab["cat"]
    assert cat_id == 0  # most frequent word gets id 0
    assert cat_count == 3


def test_preprocess_corpus_bin_drops_out_of_vocab_words(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "MIN_FREQ", 2)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    _write_corpus(raw_dir / "corpus.txt", "cat cat dog cat")

    preprocess.preprocess_corpus(raw_dir=str(raw_dir), processed_dir=str(processed_dir))

    encoded = np.fromfile(processed_dir / "corpus_index.bin", dtype=np.uint16)

    # "dog" is filtered out entirely (not kept as a placeholder), so all
    # three remaining tokens should be "cat"'s id (0).
    assert encoded.tolist() == [0, 0, 0]
