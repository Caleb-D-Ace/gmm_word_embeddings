# Project status & roadmap

See the main [README](../README.md) for how to install and use this project; this page just tracks what's done and what's deliberately deferred.

## Status

- [x] Preprocessing (`preprocess.py`) — local text/`.jsonl` files or a Hugging Face Hub dataset
- [x] Negative sampler (`sampler.py`)
- [x] Corpus dataset loader (`dataset.py`)
- [x] GMM energy model and loss functions (`gmm_word_embedding.py`)
- [x] Training loop (`gmm_training.py`) — trains end-to-end and saves `.npz`, `.pt`, and `training_state.pt`
- [x] Command-line training entry point (`main.py`)
- [x] Progressive logging — per-batch CSV with loss, energies, active-pair fraction, gradient norm, mixture-weight spread, and variance stats (see [the metrics log](TRAINING.md#metrics-log))
- [x] Checkpointing and resuming (`--checkpoint_every` / `--resume_from`, see [Checkpointing and resuming](TRAINING.md#checkpointing-and-resuming))


## Stretch goals

Ideas deliberately deferred until a faithful, working training loop exists first:

- **Multiple negative samples per positive pair.** The original paper's max-margin loss compares one true `(target, context)` pair against one sampled negative; `TrainingConfig.num_negatives` is reserved for experimenting with drawing several negatives per positive later (e.g. averaging their energies, or taking the hardest negative), once there's a working baseline to compare against.
- **Parent/child and categorical word relationships**, via the paper's own minimum-component-pair KL-divergence entailment score (its §4.7) — not yet implemented here. A small toy experiment suggested a naive "the parent has higher variance" heuristic doesn't hold on its own, but the direction of the KL divergence (child → parent lower than parent → child) does separate a parent from its children while staying roughly symmetric between co-hyponyms (e.g. "dog" vs. "cat"). That's toy-scale evidence only; it needs checking against a real hypernym dataset before being treated as more than a hypothesis.
