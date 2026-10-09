import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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


def test_id_dtype_widens_past_16_bits():
    assert preprocess.id_dtype(65535) is np.uint16
    assert preprocess.id_dtype(65536) is np.uint32


def test_large_vocab_ids_round_trip_with_id_dtype(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "MIN_FREQ", 1)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_corpus(raw_dir / "corpus.txt", " ".join(f"w{i}" for i in range(70000)))

    preprocess.preprocess_corpus(raw_dir=str(raw_dir), processed_dir=str(tmp_path / "out"), stopwords="none")

    encoded = np.fromfile(tmp_path / "out" / "corpus_index.bin", dtype=preprocess.id_dtype(70000))
    assert encoded.tolist() == list(range(70000))


def _preprocess(tmp_path, monkeypatch, text, **kwargs):
    monkeypatch.setattr(preprocess, "MIN_FREQ", 2)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    _write_corpus(raw_dir / "corpus.txt", text)
    preprocess.preprocess_corpus(raw_dir=str(raw_dir), processed_dir=str(processed_dir), **kwargs)
    vocab = json.loads((processed_dir / "sorted_vocab.json").read_text(encoding="utf-8"))
    encoded = np.fromfile(processed_dir / "corpus_index.bin", dtype=np.uint16)
    return vocab, encoded


def test_preprocess_corpus_removes_stopwords_by_default(tmp_path, monkeypatch):
    vocab, encoded = _preprocess(tmp_path, monkeypatch, "the cat the dog the cat the dog")

    assert set(vocab) == {"cat", "dog"}
    # Stopwords vanish from the encoded corpus entirely rather than leaving gaps.
    assert encoded.tolist() == [0, 1, 0, 1]


def test_preprocess_corpus_can_keep_stopwords(tmp_path, monkeypatch):
    vocab, _ = _preprocess(tmp_path, monkeypatch, "the cat the dog the cat the dog", stopwords="none")

    assert "the" in vocab


def test_preprocess_corpus_accepts_a_custom_stopword_file(tmp_path, monkeypatch):
    custom = tmp_path / "stop.txt"
    custom.write_text("cat\n", encoding="utf-8")

    vocab, _ = _preprocess(tmp_path, monkeypatch, "the cat the dog the cat the dog", stopwords=str(custom))

    assert "cat" not in vocab
    assert "the" in vocab  # only the words in the custom file are dropped


def test_jsonl_reader_treats_a_null_text_value_as_empty(tmp_path):
    # A JSON `null` deserializes to Python None, which .get(key, default) returns as-is
    # since the key IS present -- tokenize(None) used to crash on this.
    file_path = tmp_path / "sample.jsonl"
    file_path.write_text('{"text": null}\n{"text": "cat dog"}\n', encoding="utf-8")

    tokens = list(preprocess.jsonl_reader(file_path))

    assert tokens == ["cat", "dog"]


def test_parquet_reader_treats_a_null_text_value_as_empty(tmp_path):
    file_path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"text": ["cat dog", None, "bird"]}), file_path)

    tokens = list(preprocess.parquet_reader(file_path))

    assert tokens == ["cat", "dog", "bird"]


def test_parquet_reader_pulls_the_configured_text_key(tmp_path):
    file_path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"body": ["one two"], "title": ["ignored"]}), file_path)

    tokens = list(preprocess.parquet_reader(file_path, text_key="body"))

    assert tokens == ["one", "two"]


def test_corpus_streamer_reads_parquet_files(tmp_path):
    file_path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"text": ["cat dog"]}), file_path)

    tokens = sorted(preprocess.corpus_streamer(tmp_path))

    assert tokens == ["cat", "dog"]


def test_stream_file_rejects_unsupported_extensions(tmp_path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text("cat dog", encoding="utf-8")

    with pytest.raises(ValueError):
        list(preprocess.corpus_streamer(tmp_path))


def test_stream_file_rejects_compressed_parquet(tmp_path):
    # Compression wrapping only makes sense for line-oriented formats (.txt/.jsonl/.csv/.tsv);
    # .parquet already compresses internally and pyarrow doesn't read it from a stream.
    file_path = tmp_path / "sample.parquet.gz"
    file_path.write_bytes(b"not a real parquet file, just needs to exist")

    with pytest.raises(ValueError):
        list(preprocess.corpus_streamer(tmp_path))


def test_token_streamer_reads_gzip_compressed_text(tmp_path):
    import gzip

    file_path = tmp_path / "sample.txt.gz"
    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        f.write("cat dog")

    tokens = list(preprocess.token_streamer(file_path, opener=gzip.open))

    assert tokens == ["cat", "dog"]


def test_jsonl_reader_reads_bz2_compressed_jsonl(tmp_path):
    import bz2

    file_path = tmp_path / "sample.jsonl.bz2"
    with bz2.open(file_path, "wt", encoding="utf-8") as f:
        f.write('{"text": "cat dog"}\n')

    tokens = list(preprocess.jsonl_reader(file_path, opener=bz2.open))

    assert tokens == ["cat", "dog"]


def test_corpus_streamer_reads_gzip_compressed_jsonl(tmp_path):
    import gzip

    file_path = tmp_path / "sample.jsonl.gz"
    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        f.write('{"text": "cat dog"}\n')

    tokens = sorted(preprocess.corpus_streamer(tmp_path))

    assert tokens == ["cat", "dog"]


def test_csv_reader_pulls_the_configured_text_key(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("title,text\nignored,cat dog\n", encoding="utf-8")

    tokens = list(preprocess.csv_reader(file_path))

    assert tokens == ["cat", "dog"]


def test_corpus_streamer_reads_tsv_with_tab_delimiter(tmp_path):
    file_path = tmp_path / "sample.tsv"
    file_path.write_text("text\tlabel\ncat dog\tpositive\n", encoding="utf-8")

    tokens = sorted(preprocess.corpus_streamer(tmp_path))

    assert tokens == ["cat", "dog"]


def test_csv_reader_treats_a_missing_value_as_empty(tmp_path):
    # An empty CSV cell comes back as '' already, but a row shorter than the header gives
    # DictReader's missing-column default (None) -- same null-safety case as the other readers.
    file_path = tmp_path / "sample.csv"
    file_path.write_text("text,label\ncat dog,positive\n,missing\n", encoding="utf-8")

    tokens = list(preprocess.csv_reader(file_path))

    assert tokens == ["cat", "dog"]


def test_arrow_reader_reads_the_file_format(tmp_path):
    import pyarrow as pa

    file_path = tmp_path / "sample.arrow"
    table = pa.table({"text": ["cat dog", None, "bird"]})
    with pa.OSFile(str(file_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    tokens = list(preprocess.arrow_reader(file_path))

    assert tokens == ["cat", "dog", "bird"]


def test_arrow_reader_reads_the_stream_format(tmp_path):
    import pyarrow as pa

    file_path = tmp_path / "sample.arrow"
    table = pa.table({"text": ["one two"]})
    with pa.OSFile(str(file_path), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

    tokens = list(preprocess.arrow_reader(file_path))

    assert tokens == ["one", "two"]


def test_corpus_streamer_reads_arrow_files(tmp_path):
    import pyarrow as pa

    file_path = tmp_path / "sample.arrow"
    table = pa.table({"text": ["cat dog"]})
    with pa.OSFile(str(file_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    tokens = sorted(preprocess.corpus_streamer(tmp_path))

    assert tokens == ["cat", "dog"]


def test_corpus_streamer_skips_hidden_files(tmp_path):
    # .gitkeep and similar dotfiles are filesystem/git bookkeeping, not corpus content --
    # they shouldn't be tokenized, and (since they have no recognized extension) shouldn't
    # raise either.
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    _write_corpus(tmp_path / "corpus.txt", "cat dog")

    tokens = sorted(preprocess.corpus_streamer(tmp_path))

    assert tokens == ["cat", "dog"]
