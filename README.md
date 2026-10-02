# gmm_word_embeddings

A modern PyTorch implementation of [Gaussian Mixture Word Embeddings based on Athiwaratkun and Wilson (ACL 2017)](https://arxiv.org/abs/1704.08424).

This repository is an offline training pipeline that turns a raw text corpus into a language model of multi-sense, probabilistic word distributions: instead of one vector per word, each word is represented as a Gaussian mixture over the embedding space, so words with multiple senses (e.g. "bank") can occupy multiple distinct modes.


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

   This produces `data/processed/sorted_vocab.json` and `data/processed/corpus_index.bin`. Run `python src/preprocess.py --help` for the full list of options (custom `--raw_dir`/`--processed_dir`, `--text_key`, `--hf_split`, `--stopwords`, etc.). Re-run it whenever you change the corpus or the stopword setting, since the vocabulary changes with them.

5. Run training from the project root:

   ```bash
   # With default parameters (reads data/processed/, writes data/model/)
   python main.py

   # Or override any of them
   python main.py --embedding_dim 100 --k 3 --epochs 100 --lr 0.02
   ```

   Every flag has a default, so a bare `python main.py` works once `data/processed/` exists from the preprocessing step. Run `python main.py --help` for the full list. Here are the most notable ones:

   | Flag | Default | Meaning |
   |---|---|---|
   | `--processed_dir` | `data/processed` | Where to read `sorted_vocab.json`/`corpus_index.bin` from |
   | `--output_path` | `data/model` | Directory to write the trained model into |
   | `--embedding_dim` | `50` | Dimensionality (`D`) of each Gaussian component |
   | `--k` | `2` | Number of mixture components (`K`) per word |
   | `--window_size` | `5` | Context window size on each side of the center word (counted after stopword removal) |
   | `--batch_size` | `256` | Training batch size |
   | `--epochs` | `50` | Number of training epochs |
   | `--optimizer` | `adam` | `adam` or `adagrad` |
   | `--lr` | `0.01` | Initial learning rate |
   | `--lr_final` | `1e-5` | Learning rate at the end of training; it decays linearly from `--lr` over the whole run |
   | `--margin` | `1.0` | Margin for the max-margin ranking loss |
   | `--checkpoint_every` | `0` | Save a snapshot model  into `output_path/epoch_<N>/` every N epochs (`0` disables it) |
   | `--log_dir` | `logs/` | Where the per-batch training log CSV is written |
   | `--num_negatives` | `1` | Negative samples per positive pair — **don't change this yet**; multi-negative training isn't implemented (see stretch goals) |

   This produces `data/model/gmm_embeddings.npz` and `data/model/gmm_embeddings.pt`.


## How it works

Training happens in two independent stages:

1. **Preprocessing** (`src/preprocess.py`) — tokenizes a corpus from either local files or a Hugging Face dataset, drops stopwords, and builds a vocabulary of every remaining word occurring at least `MIN_FREQ` times (40 by default — currently a constant at the top of the file, not yet a command-line option). Stopwords come from a built-in English list by default and can be changed with `--stopwords` (see the differences above). The corpus can come from either of two sources:
   - **Local files** (default) — every file under `--raw_dir` (default `data/raw/`), searched recursively. Plain text files are tokenized directly; `.jsonl` files have a configurable field (`--text_key`, default `text`) pulled out of each line first, so a pre-downloaded dump of JSON records works without conversion.
   - **A Hugging Face Hub dataset** (`--hf_dataset`, e.g. `wikimedia/wikipedia`, with optional `--hf_config`/`--hf_split`) — downloaded and cached by the [`datasets`](https://pypi.org/project/datasets/) library instead of being saved into `data/raw/`. `datasets` is an optional dependency, only needed for this path.

   Either way, it writes:
   - `data/processed/sorted_vocab.json` — a single mapping of `word -> (id, frequency count)`, ordered by descending frequency, so a word's position in the file is also its integer id.
   - `data/processed/corpus_index.bin` — the entire corpus re-encoded as a flat binary array of those integer ids (2 bytes each, or 4 when the vocabulary exceeds 65,535 words), for fast memory-mapped reading during training.

2. **Training** (`main.py` / `src/gmm_training.py`) — loads the vocabulary and binary corpus, then trains a `GMMWordEmbedding` model: each word is represented by `K` Gaussian components (mean, variance, and mixture weight) in a `D`-dimensional embedding space. Training uses a skip-gram-style objective — for each (target word, real context word) pair, a negative context word is sampled by unigram frequency^0.75 (as in word2vec), and a max-margin ranking loss pushes the real pair's Gaussian-mixture "energy" (expected likelihood overlap) above the negative pair's. It optimizes with Adam, with a learning rate that decays linearly over the whole run. The learned parameters are saved to both `gmm_embeddings.npz` (raw numpy arrays — `mu`, `var`, `mix_weights` — usable from any language) and `gmm_embeddings.pt` (a PyTorch `state_dict`, for loading straight back into `GMMWordEmbedding` in Python).

## How to use the resulting word embedding

Training writes three files into `data/model/` (or wherever `--output_path` pointed):

- **`sorted_vocab.json`** — the `word -> [id, frequency count]` mapping needed to interpret the other two files.
- **`gmm_embeddings.pt`** — a PyTorch `state_dict`, for loading straight back into `GMMWordEmbedding` in Python.
- **`gmm_embeddings.npz`** — a compressed NumPy archive with the same trained values as plain arrays, readable from any language.

You'll always need `sorted_vocab.json` alongside whichever model file you use, since word ids only mean something in the context of that mapping. Which model file to select depends on your environment:

- **Python + PyTorch** — use `gmm_embeddings.pt`.
- **Anything else** (plain NumPy, or another language entirely) — use `gmm_embeddings.npz`. You'll need a library capable of reading the `.npz` zip-of-arrays format for your language, e.g. NumPy itself in Python, or [`cnpy`](https://github.com/rogersce/cnpy) in C++. Search for a library in your language.

### The data's shape

Every array is indexed by word id — the same integer assigned in `sorted_vocab.json`, and the same row order used to size the model when it was trained. Row `i` in every array corresponds to word id `i`:

- **`mu`** — shape `(vocab_size, K, D)`. `mu[i]` gives word `i`'s `K` Gaussian component means, each a `D`-dimensional vector — up to `K` separate "sense locations" for that word.
- **`var`** — shape `(vocab_size, K, D)`. `var[i]` gives the spread of each of those `K` components, per dimension (a diagonal covariance — each dimension is independent within a component).
- **`mix_weights`** — shape `(vocab_size, K)`, each row summing to 1. `mix_weights[i]` gives how much probability mass each of word `i`'s `K` components carries. A dominant weight (e.g. `[0.95, 0.05]`) means the word is effectively single-sense; a more even split (e.g. `[0.55, 0.45]`) means the model is genuinely using both components — this is how it represents polysemy (e.g. "bank" splitting into a financial-institution sense and a riverbank sense), which a standard word2vec-style embedding can't do.

### Looking up a word

```python
import json
import numpy as np

with open("data/model/sorted_vocab.json") as f:
    vocab = json.load(f)

data = np.load("data/model/gmm_embeddings.npz")
mu, var, mix_weights = data["mu"], data["var"], data["mix_weights"]

word_id, _count = vocab["bank"]
print(mu[word_id])           # shape (K, D) — "bank"'s K component means
print(mix_weights[word_id])  # shape (K,)   — how much weight each sense carries
```

### Comparing two words

Every word has `K` component means rather than a single vector, so a plain cosine similarity or dot product between two "word vectors" isn't defined. The original paper (Table 2) evaluates three ways of scoring a pair of words, each of which looks across component pairs:

- **Maximum cosine similarity** — the highest cosine similarity between any component of word A and any component of word B, using only the means. The paper found this the best-performing measure on most word-similarity benchmarks, and its nearest-neighbor examples use it too.
- **Minimum Euclidean distance** between any pair of component means. It did better than maximum cosine on a couple of datasets.
- **Expected likelihood energy** — the function training itself optimizes. It also uses `var` and `mix_weights`, so it accounts for how uncertain each sense is. It scored below maximum cosine on most of the paper's similarity benchmarks, but it is the score the model was actually trained to make high for related words.

#### Maximum cosine similarity

Only `mu` is needed, so this works from the `.npz` in any language:

```python
def max_cosine(mu, a, b):
    """Highest cosine similarity between any component of word id a and any component of word id b."""
    A = mu[a] / np.linalg.norm(mu[a], axis=1, keepdims=True)   # (K, D), unit-length rows
    B = mu[b] / np.linalg.norm(mu[b], axis=1, keepdims=True)
    return (A @ B.T).max()

max_cosine(mu, vocab["bank"][0], vocab["river"][0])
```

#### Energy from Python

A `state_dict` is just a dict of tensors — it can only be loaded into an already-constructed instance of the matching class, so you'll need `GMMWordEmbedding`'s class definition available in your own project (copy `src/gmm_word_embedding.py` in, or depend on this repo directly):

```python
import json
import torch
from gmm_word_embedding import GMMWordEmbedding  # copied from src/

with open("data/model/sorted_vocab.json") as f:
    vocab = json.load(f)

# embedding_dim/K must match exactly what you trained with — a mismatch makes
# load_state_dict fail, since the saved tensor shapes won't line up otherwise.
model = GMMWordEmbedding(vocab_size=len(vocab), embedding_dim=50, K=2)
model.load_state_dict(torch.load("data/model/gmm_embeddings.pt"))
model.eval()

bank_id, river_id = vocab["bank"][0], vocab["river"][0]
with torch.no_grad():
    energy = model.forward(torch.tensor([bank_id]), torch.tensor([river_id]))
```

No need to reimplement `log_overlap`/`gmm_energy` yourself — call them, or just `forward`, directly.

#### Energy from `.npz`, or another language

You only have the raw `mu`/`var`/`mix_weights` arrays here, so the energy function has to be implemented yourself. For two words with means/variances `(mu1, var1)` and `(mu2, var2)` (each shape `(K, D)`) and mixture weights `w1`, `w2` (each shape `(K,)`):

1. **Pairwise component log-overlap** — for every pair of components `(k, m)`, one from each word, the expected likelihood kernel between those two Gaussians:

   ```
   log_overlap[k, m] = -0.5 * sum_d( log(var1[k,d] + var2[m,d]) + (mu1[k,d] - mu2[m,d])^2 / (var1[k,d] + var2[m,d]) )
                        - (D / 2) * log(2 * pi)
   ```

2. **Combine into a single energy score**, weighting each component pair by how much mixture weight each side gives it:

   ```
   energy = log( sum_k sum_m( w1[k] * w2[m] * exp(log_overlap[k, m]) ) )
   ```

   Compute this in log space with a `logsumexp` (`energy = logsumexp_{k,m}( log(w1[k]) + log(w2[m]) + log_overlap[k, m] )`) rather than calling `exp` directly. The log-overlaps are large negative numbers (around -60 at `D=50`), and `exp` of those is tiny enough that adding any small epsilon before the `log` would swamp the real value and flatten every energy to the same number.

Higher `energy` means more similar. This is exactly `GMMWordEmbedding.log_overlap`/`gmm_energy` in `src/gmm_word_embedding.py`, if you want to cross-check a port against the reference implementation.


### Differences from the original implementation

- Uses PyTorch instead of TensorFlow to train the word embedding model.
- Outputs a `.npz` file (optimized for NumPy), so the trained model is technically language-agnostic to load (C++ via `cnpy`, Rust via `npy-rs`, etc.).
- Intended to eventually capture parent/child and categorical relationships between words, in addition to similarity.
- **Optimizer: Adam instead of Adagrad.** The paper chose Adagrad because Adam was about 10x slower in its TensorFlow implementation, which is a property of that setup rather than an argument about quality. In small toy runs, on a corpus where one word appears in two unrelated topics, Adam separated that word's two senses much faster. The default learning rate (0.01, decaying linearly to 1e-5 over the whole run, as in the paper) was tuned on those toy runs only, so expect to retune it for a real corpus, and on a very large one the comparison with Adagrad may differ. Adagrad remains available with `--optimizer adagrad`, which wants a larger learning rate such as 0.05.
- **Stopwords removed instead of subsampled.** The paper discards frequent words probabilistically (frequency-based subsampling). This project drops stopwords outright during preprocessing, so function words like "the" can't crowd out the neighbors of content words. The built-in English list is deliberately conservative: function words only, because some popular lists also drop content words such as "fire" or "bill". Use `--stopwords none` to keep everything, or `--stopwords path/to/list.txt` (one word per line) for another language or a custom list. Because stopwords are dropped before the corpus is encoded, the words on either side of one become neighbors, so a context window of 5 covers roughly what the paper's window of 10 covers in raw text.
- **One shared embedding table.** The paper trains separate input and output distributions for each word and keeps only the input ones. Here a word uses the same distribution as a target and as a context. That halves the parameters, and the energy between two words is exactly the score training optimizes, which suits scoring association. The cost is that a word is scored against itself as a maximally similar match, and a word occasionally drawn as its own negative pulls on its variance; separate tables avoid both and tend to favor "interchangeable word" similarity over "associated word" similarity.
- **Other defaults that differ:** diagonal rather than spherical covariance (the paper's experiments used spherical); variances kept positive with softplus rather than optimizing their log; batch size 256 rather than 128; and initialization, where variances start near 1.0 and means near ±0.5/D, versus the paper's 0.05 and ±√(3/D). In toy tests the paper's initialization split a two-sense word's components in 10 of 10 seeds against 3 of 10 for this one, so the initialization is under review.


## Project status

- [x] Preprocessing (`preprocess.py`) — local text/`.jsonl` files or a Hugging Face Hub dataset
- [x] Negative sampler (`sampler.py`)
- [x] Corpus dataset loader (`dataset.py`)
- [x] GMM energy model and loss functions (`gmm_word_embedding.py`)
- [x] Training loop (`gmm_training.py`) — trains end-to-end and saves both `.npz` and `.pt`
- [x] Command-line training entry point (`main.py`)
- [ ] Progressive logging





## Stretch goals

Ideas deliberately deferred until a faithful, working training loop exists first:

- **Multiple negative samples per positive pair.** The original paper's max-margin loss compares one true `(target, context)` pair against one sampled negative; `TrainingConfig.num_negatives` is reserved for experimenting with drawing several negatives per positive later (e.g. averaging their energies, or taking the hardest negative), once there's a working baseline to compare against.
- Parent/child and categorical word relationships (see "Differences from the original implementation" above).