import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Union

"""
Preprocess.py prepares a corpus for training.
    - Tokenizes every word in the corpus
    - Creates a frequency count for each token
    - Indexes all words of a frequency above MIN_FREQ and creates word_to_id and id_to_word for easy index/word translation
    - Outputs corpus_index.bin, containing the entire corpus represented as integer indices 
"""
MIN_FREQ = 40   # The minimum occurrence amount required in order to be considered a part of the vocabulary

# Method to stream a file word-by-word, yielding cleaned tokens as it goes
def token_streamer(file_path: Path) -> Iterator[str]:
    word_regex = re.compile(r'\w+')
    with open(file_path, 'r', encoding = 'utf-8', errors='ignore') as file:
        for line in file:
            for match in word_regex.finditer(line):
                yield match.group().lower()

# Method to run through every file in a corpus stored in data/raw/ and yield the tokens contained therein. 
def corpus_streamer(input_path: Union[str, Path]):
    path = Path(input_path)

    if path.is_file():
        yield from token_streamer(path)
    elif path.is_dir():
        # rglob('*') recursively finds all files inside subfolders of 'path'
        for file_path in path.rglob('*'):
            if file_path.is_file():
                yield from token_streamer(file_path)
    else:
        raise FileNotFoundError(f"Corpus input path {input_path} does not exist.")
    
# Method to turn a corpus of data into an indexed representation of the corpus and save the resulting word/id mappings as json files for later use.
def preprocess_corpus(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    # Establish directories
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Make a counter object to hold the streamed corpus. This allows us to cull infrequent words.
    raw_freq = Counter(corpus_streamer(raw_path))

    # Build the vocab mappings from the frequency chart...
    filtered_freq: list[str] = [word for word, count in raw_freq.most_common() if count >= MIN_FREQ]
    word_to_id: Dict[str, int] = {word: rank for rank, word in enumerate(filtered_freq)}
    id_to_word: Dict[int, str] = {rank: word for rank, word in enumerate(filtered_freq)}
    # ...and then store the indexes in json format.
    with open(processed_path / "word_to_id.json", "w", encoding="utf-8") as f:
        json.dump(word_to_id, f, ensure_ascii=False)
    with open(processed_path / "id_to_word.json", "w", encoding="utf-8") as f:
        json.dump(id_to_word, f, ensure_ascii=False)

    use_uint16: bool = len(word_to_id) <= 65535
    encoding_width = 'H' if use_uint16 else 'I'

    # Write data to binary file
    out_file = processed_path / "corpus_index.bin"
    with open(out_file, "wb") as f_out:
        for word in token_streamer(raw_path):
            word_id = word_to_id.get(word)
            if word_id is not None:
                # Pack the integer into binary and store it
                f_out.write(struct.pack(encoding_width, word_id))
    print(f"Preprocessing: Corpus saved as a binary file to {out_file}.")

# Ensures code doesn't automatically run when imported into another file
if __name__ == "__main__":
    preprocess_corpus(raw_dir="data/raw", processed_dir="data/processed")


