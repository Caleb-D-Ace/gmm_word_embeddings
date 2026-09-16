# gmm_word_embeddings

A modern PyTorch implementation of [Gaussian Mixture Word Embeddings based on Athiwaratkun and Wilson (ACL 2017)](https://arxiv.org/abs/1704.08424).

This repository is an offline training pipeline that turns a raw text corpus into a language model of multi-sense, probabilistic word distributions: instead of one vector per word, each word is represented as a Gaussian mixture over the embedding space, so words with multiple senses (e.g. "bank") can occupy multiple distinct modes.

### Differences from the original implementation

- Uses PyTorch instead of TensorFlow to train the word embedding model.
- Outputs a `.npz` file (optimized for NumPy), so the trained model is technically language-agnostic to load (C++ via `cnpy`, Rust via `npy-rs`, etc.).
- Intended to eventually capture parent/child and categorical relationships between words, in addition to similarity.

## How it works

Training happens in two independent stages:

1. **Preprocessing** (`src/preprocess.py`) — tokenizes a corpus from one of two sources and builds a vocabulary of every word occurring at least `MIN_FREQ` times (40 by default — currently a constant at the top of the file, not yet a command-line option):
   - **Local files** (default) — every file under `--raw_dir` (default `data/raw/`), searched recursively. Plain text files are tokenized directly; `.jsonl` files have a configurable field (`--text_key`, default `text`) pulled out of each line first, so a pre-downloaded dump of JSON records works without conversion.
   - **A Hugging Face Hub dataset** (`--hf_dataset`, e.g. `wikimedia/wikipedia`, with optional `--hf_config`/`--hf_split`) — downloaded and cached by the [`datasets`](https://pypi.org/project/datasets/) library instead of being saved into `data/raw/`. `datasets` is an optional dependency, only needed for this path.

   Either way, it writes:
   - `data/processed/sorted_vocab.json` — a single mapping of `word -> (id, frequency count)`, ordered by descending frequency, so a word's position in the file is also its integer id.
   - `data/processed/corpus_index.bin` — the entire corpus re-encoded as a flat binary array of those integer ids, for fast memory-mapped reading during training.

2. **Training** (`main.py` / `src/gmm_training.py`) — loads the vocabulary and binary corpus, then trains a `GMMWordEmbedding` model: each word is represented by `K` Gaussian components (mean, variance, and mixture weight) in a `D`-dimensional embedding space. Training uses a skip-gram-style objective — for each (target word, real context word) pair, a negative context word is sampled by unigram frequency^0.75 (as in word2vec), and a max-margin ranking loss pushes the real pair's Gaussian-mixture "energy" (expected likelihood overlap) above the negative pair's. The learned parameters are saved to a `.npz` file.

## Project status

- [x] Preprocessing (`preprocess.py`) — local text/`.jsonl` files or a Hugging Face Hub dataset
- [x] Negative sampler (`sampler.py`)
- [x] Corpus dataset loader (`dataset.py`)
- [x] GMM energy model and loss functions (`gmm_word_embedding.py`)
- [ ] Training loop (`gmm_training.py`) — vocabulary loading and pipeline setup are done; the training loop itself and saving the trained model are still in progress
- [ ] Command-line training entry point (`main.py`) — argument parsing is in place, but not usable end-to-end until the training loop above is finished

Command-line usage for training will be documented here once `main.py` is complete.

## Installation

1. Install PyTorch matching your hardware. Only install the CUDA build if you have an NVIDIA GPU — the CPU build will run much slower.

   ```bash
   # For CUDA 12.1 (NVIDIA GPU)
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # For CPU-only
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Choose a corpus source:
   - **Bring your own corpus** — place text or `.jsonl` files (any subfolder structure) inside `data/raw/`.
   - **Or download one from Hugging Face instead** — skip this step and pass `--hf_dataset` in step 4.

4. Run preprocessing from the project root:

   ```bash
   # Using your own local corpus in data/raw/
   python src/preprocess.py

   # Or downloading a dataset from Hugging Face instead
   python src/preprocess.py --hf_dataset wikimedia/wikipedia --hf_config 20231101.simple
   ```

   This produces `data/processed/sorted_vocab.json` and `data/processed/corpus_index.bin`. Run `python src/preprocess.py --help` for the full list of options (custom `--raw_dir`/`--processed_dir`, `--text_key`, `--hf_split`, etc.).

Training instructions will be added once `main.py` is finished.
