import pickle
from keras.preprocessing.text import Tokenizer
from env_config import *


class LlmTokenizer:
    def __init__(self):
        # Character-level tokenizer
        self.tokenizer = Tokenizer(char_level=True, filters='', lower=True)

    def encode(self, s):
        return self.tokenizer.texts_to_sequences([s])[0]

    def decode(self, l):
        return self.tokenizer.sequences_to_texts([l])[0]

    def save_obj(self):
        """
        Saves the tokenizer object to file
        """
        with open(TOK_OBJ_PATHNAME, 'wb') as f:
            pickle.dump(self, f)
            print(f"\nLLM tokenizer object has been saved to {TOK_OBJ_PATHNAME}")

    def save_tok(self):
        """
        Saves the tokenizer to file
        """
        with open(TOK_PATHNAME, 'wb') as f:
            pickle.dump(self.tokenizer, f)
            print(f"LLM tokenizer has been saved to {TOK_PATHNAME}")

    @staticmethod
    def load_tok():
        with open(TOK_PATHNAME, 'rb') as f:
            return pickle.load(f)

