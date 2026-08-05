# gmm_word_embeddings

A modern PyTorch implementation of Gaussian Mixture Word Embeddings based on Athiwaratkun and Wilson (ACL 2017).

This repository is an offline training pipeline to create a language model from raw text corpora containing multi-sense, probablistic word distributions.


Differences from their original implementation: 
- Uses PyTorch instead of TensorFlow to train the word embedding model
- Outputs a .npz file (optimized for numpy), meaning the model is technically language-agnostic (C++: cnpy, Rust: npy-rs, etc.)
- Intended to provide a language model that, in addition to containing word similarity information, learns parent/child and categorical relationships between words.


### Installation

1. Install PyTorch matching your hardware. Only download the CUDA version if you have an NVIDIA GPU. The CPU version will run much slower.
   ```bash
   # For CUDA 12.1 (NVIDIA GPU) 
   pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
   

   # For CPU-only
   pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)