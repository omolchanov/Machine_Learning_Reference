import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np

from env_config import *
from llm_corpus import LlmCorpus
from llm_tokenizer import LlmTokenizer


class LlmDataset:
    def __init__(self):
        self.tokenizer_obj = LlmTokenizer()
        self.tokenizer = self.tokenizer_obj.tokenizer

        self.vocab_size = None

        self.ds = None
        self.metadata = None

    def get_dataset(self):
        corpus = LlmCorpus.get_corpus()

        # Tokenize the data
        self.tokenizer.fit_on_texts([corpus])
        self.vocab_size = len(self.tokenizer.word_index) + 1  # add 1 for padding index (0)
        print('\nVocalbulary size:', self.vocab_size)

        # Encode entire dataset
        self.ds = np.array(self.tokenizer_obj.encode(corpus), dtype=np.int32)
        print('Encoded dataset shape:', self.ds.shape[0])

        self.tokenizer_obj.save_obj()
        self.tokenizer_obj.save_tok()

        self._save_ds()
        self._save_metadata()

    @staticmethod
    def load_ds():
        return np.load(DS_PATHNAME)

    def _save_ds(self):
        """
        Saves the dataset to file
        """
        np.save(DS_PATHNAME, self.ds)
        print(f"Dataset has been saved to {DS_PATHNAME}")

    def _save_metadata(self):
        """
        Saves the dataset's metadata to file
        """
        metadata = {
            'dataset_size': DATASET_SIZE * 3,  # 3 datasets used for training
            'vocab_size': self.vocab_size
        }

        with open(DS_MD_PATHNAME, 'w') as f:
            json.dump(metadata, f, indent=2)
            print(f"Metadata {metadata} has been saved to {DS_MD_PATHNAME}")


if __name__ == '__main__':
    llm_ds = LlmDataset()
    llm_ds.get_dataset()
