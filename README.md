# gmm_word_embeddings

A modern PyTorch implementation of [Gaussian Mixture Word Embeddings based on Athiwaratkun and Wilson (ACL 2017)](https://arxiv.org/abs/1704.08424).

This repository is an offline training pipeline that turns a raw text corpus into a language model of multi-sense, probabilistic word distributions: instead of one vector per word, each word is represented as a Gaussian mixture over the embedding space, so words with multiple senses (e.g. "bank") can occupy multiple distinct modes.

### Differences from the original implementation

- Uses PyTorch instead of TensorFlow to train the word embedding model.
- Outputs a `.npz` file (optimized for NumPy), so the trained model is technically language-agnostic to load (C++ via `cnpy`, Rust via `npy-rs`, etc.).
- Intended to eventually capture parent/child and categorical relationships between words, in addition to similarity.

## How it works

Training happens in two independent stages:

1. **Preprocessing** (`src/preprocess.py`) — reads every file under `data/raw/`, tokenizes it, and builds a vocabulary of every word occurring at least `MIN_FREQ` times (40 by default — currently a constant at the top of the file, not yet a command-line option). It writes:
   - `data/processed/sorted_vocab.json` — a single mapping of `word -> (id, frequency count)`, ordered by descending frequency, so a word's position in the file is also its integer id.
   - `data/processed/corpus_index.bin` — the entire corpus re-encoded as a flat binary array of those integer ids, for fast memory-mapped reading during training.

2. **Training** (`main.py` / `src/gmm_training.py`) — loads the vocabulary and binary corpus, then trains a `GMMWordEmbedding` model: each word is represented by `K` Gaussian components (mean, variance, and mixture weight) in a `D`-dimensional embedding space. Training uses a skip-gram-style objective — for each (target word, real context word) pair, a negative context word is sampled by unigram frequency^0.75 (as in word2vec), and a max-margin ranking loss pushes the real pair's Gaussian-mixture "energy" (expected likelihood overlap) above the negative pair's. The learned parameters are saved to a `.npz` file.

## Project status

- [x] Preprocessing (`preprocess.py`)
- [x] Negative sampler (`sampler.py`)
- [x] Corpus dataset loader (`dataset.py`)
- [x] GMM energy model and loss functions (`gmm_word_embedding.py`)
- [ ] Training loop (`gmm_training.py`) — in progress
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

3. Place your raw corpus (one or more text files, in any subfolder structure) inside `data/raw/`.

4. Run preprocessing from the project root:

   ```bash
   python src/preprocess.py
   ```

   This produces `data/processed/sorted_vocab.json` and `data/processed/corpus_index.bin`.

Training instructions will be added once `main.py` is finished.
