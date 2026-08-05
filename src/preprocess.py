import io
import re

"""
Preprocess prepares a corpus for training.
    - Tokenizes every word in the corpus
    - Creates a frequency count for each token
    - Indexes all words of a frequency above MIN_FREQ and creates word_to_id and id_to_word for easy index/word translation
    - Outputs corpus_ids.bin, containing the entire corpus represented as integer indices 
"""
class Preprocess():
    MIN_FREQ = 40   # The minimum occurrence amount required in order to be considered a part of the vocabulary

    def tokenize(raw_word: str) -> str:
        return raw_word.lower().strip()