import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import json

from keras.preprocessing.text import Tokenizer
from datasets import load_dataset

import numpy as np


DATASET_SIZE = 200

DATA_DIRECTORY_PATH = 'data'
TOKENIZER_FILENAME = 'tokenizer.pkl'
DS_FIlENAME = 'dataset.npy'
DS_METADATA_FILENAME = 'ds_metadata.json'


class LlmTokenizer:
    def __init__(self):

        # Character-level tokenizer
        self.tokenizer = Tokenizer(char_level=True, filters='', lower=True)

    def encode(self, s):
        return self.tokenizer.texts_to_sequences([s])[0]

    def decode(self, l):
        return self.tokenizer.sequences_to_texts([l])[0]

    def tokenize_and_get_dataset(self):

        # Load wikitext-2-raw-v1 from Hugging Face
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        print('The dataset is loaded')

        # Combine all lines into one big string (truncate to 50k chars for speed)
        raw_text = "\n".join(dataset["text"]).lower()[:DATASET_SIZE]

        # Tokenize the data
        self.tokenizer.fit_on_texts([raw_text])
        vocab_size = len(self.tokenizer.word_index) + 1  # add 1 for padding index (0)

        # Encode entire dataset
        data = np.array(self.encode(raw_text), dtype=np.int32)
        print('Encoded dataset shape:', data.shape[0])

        # Save the tokenizer
        with open(f"{DATA_DIRECTORY_PATH}/{TOKENIZER_FILENAME}", 'wb') as f:
            pickle.dump(self.tokenizer, f)

        # Save the dataset
        np.save(f"{DATA_DIRECTORY_PATH}/{DS_FIlENAME}", data)

        # Save metadata to JSON
        metadata = {
            'dataset_size': DATASET_SIZE,
            'vocab_size': vocab_size
        }

        with open(f"{DATA_DIRECTORY_PATH}/{DS_METADATA_FILENAME}", 'w') as f:
            json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    llm_tok = LlmTokenizer()
    llm_tok.tokenize_and_get_dataset()
