import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import json

from keras.preprocessing.text import Tokenizer
from datasets import load_dataset

import numpy as np


DATASET_SIZE = int(6 * 10e2)

DATA_DIRECTORY_PATH = 'data'
TOKENIZER_FILENAME = 'tokenizer.pkl'
TOKENIZER_OBJ_FILENAME = 'tokenizer_obj.pkl'
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

        # Load Books Corpus dataset from Hugging Face
        books_corp_ds = load_dataset('bookcorpus', split=f'train[:{DATASET_SIZE}]', trust_remote_code=True)
        print('The Book Corpus dataset is loaded')

        # Load wikitext-2-raw-v1 from Hugging Face
        wikitext_ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=f'train[:{DATASET_SIZE}]')
        print('The Wiki dataset is loaded')

        # Load WikiText-103
        wikitext103_ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=f'train[:{DATASET_SIZE}]')
        print('WikiText-103 dataset loaded')

        # Combine all lines into one big string (truncate to DATASET_SIZE chars for speed)
        books_corp_text = "\n".join(books_corp_ds["text"]).lower()
        wikitext_text = "\n".join(wikitext_ds["text"]).lower()
        wikitext103_text = "\n".join(wikitext103_ds["text"]).lower()

        # Combine all datasets
        combined_text = books_corp_text + "\n" + wikitext_text + "\n" + wikitext103_text

        # Tokenize the data
        self.tokenizer.fit_on_texts([combined_text])
        vocab_size = len(self.tokenizer.word_index) + 1  # add 1 for padding index (0)
        print('\nVocalbulary size:', vocab_size)

        # Encode entire dataset
        data = np.array(self.encode(combined_text), dtype=np.int32)
        print('Encoded dataset shape:', data.shape[0])

        # Save the LLM tokenizer object
        pathname = f"{DATA_DIRECTORY_PATH}/{TOKENIZER_OBJ_FILENAME}"
        with open(pathname, 'wb') as f:
            pickle.dump(self, f)
            print(f"\nLLM tokenizer object has been saved to {pathname}")

        # Save the tokenizer
        pathname = f"{DATA_DIRECTORY_PATH}/{TOKENIZER_FILENAME}"
        with open(pathname, 'wb') as f:
            pickle.dump(self.tokenizer, f)
            print(f"LLM tokenizer has been saved to {pathname}")

        # Save the dataset
        pathname = f"{DATA_DIRECTORY_PATH}/{DS_FIlENAME}"
        np.save(pathname, data)
        print(f"Dataset has been saved to {pathname}")

        # Save metadata to JSON
        metadata = {
            'dataset_size': DATASET_SIZE * 3,  # 3 datasets used for training
            'vocab_size': vocab_size
        }

        pathname = f"{DATA_DIRECTORY_PATH}/{DS_METADATA_FILENAME}"
        with open(pathname, 'w') as f:
            json.dump(metadata, f, indent=2)
            print(f"Metadata {metadata} has been saved to {pathname}")


if __name__ == '__main__':
    llm_tok = LlmTokenizer()
    llm_tok.tokenize_and_get_dataset()
