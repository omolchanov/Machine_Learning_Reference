import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

from llm_tokenizer import LlmTokenizer
from llm_dataset import LlmDataset
from llm_model_entity import LlmModelEntity
from env_config import *

MODEL_ID = 'llm_model_546_30'


class LlmChat:

    def __init__(self):
        self.tokenizer = LlmTokenizer.load_tok()
        self.data = LlmDataset.load_ds()
        self.model = LlmModelEntity.load(MODEL_ID)
        self.m_metadata = LlmModelEntity.load_metadata(MODEL_ID)

        self.block_size = self.m_metadata.get('block_size')

    def generate_text(self, prompt, length=ANSWER_LENGTH, temperature=1.0):
        block_size = self.block_size

        input_ids = self.tokenizer.texts_to_sequences([prompt])[0]
        if not input_ids:
            return f"Error: Could not tokenize '{prompt}'"

        generated = input_ids[:]
        context = generated[-block_size:]

        for _ in range(length):
            # Pad context if needed
            if len(context) < block_size:
                context = [0] * (block_size - len(context)) + context
            else:
                context = context[-block_size:]

            input_tensor = tf.constant([context], dtype=tf.int32)
            logits = self.model(input_tensor, training=False)

            # Use sampling or deterministic argmax decoding
            next_token = tf.random.categorical(logits[0, -1:] / temperature, 1).numpy()[0][0]

            generated.append(next_token)
            context.append(next_token)

        # Decode generated tokens to text
        try:
            return self.tokenizer.sequences_to_texts([generated])[0]
        except Exception:
            return ' '.join([self.tokenizer.index_word.get(i, '<UNK>') for i in generated if i > 0])


if __name__ == '__main__':
    chat = LlmChat()

    while True:
        prompt = input('> ')

        if prompt == 'q!':
            print('Bye!')
            exit(0)

        response = chat.generate_text(prompt)
        print(response)
