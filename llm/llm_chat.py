import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import pickle

import tensorflow as tf
import numpy as np

from llm_tokenizer import (
    DATA_DIRECTORY_PATH,
    DS_FIlENAME,
    TOKENIZER_FILENAME
)
from llm_model import (
    MODELS_DIRECTORY_PATH,
    MODEL_METADATA_FILENAME
)


# Load tokenizer
with open(f"{DATA_DIRECTORY_PATH}/{TOKENIZER_FILENAME}", 'rb') as f:
    tokenizer = pickle.load(f)

# Load dataset
data = np.load(f"{DATA_DIRECTORY_PATH}/{DS_FIlENAME}")

# Load the model
model_id = 'llm_model_550_0.107_200'
model = tf.keras.models.load_model(f"{MODELS_DIRECTORY_PATH}/{model_id}")

print(f"Model {model_id} has been loaded")

# Load the model's metadata
with open(f"{MODELS_DIRECTORY_PATH}/{model_id}/{MODEL_METADATA_FILENAME}", 'r') as f:
    m_metadata = json.load(f)

    print(f"The Model's metadata has been loaded\n{m_metadata}\n")

    # Reading the required model's hyperparameters
    block_size = m_metadata.get('block_size')


def generate_text(prompt, length=200, temperature=1.0, block_size=block_size):
    try:
        input_ids = tokenizer.texts_to_sequences([prompt])[0]
        if not input_ids:
            return f"Error: Could not tokenize '{prompt}'"

        generated = input_ids[:]
        print(f"Starting with tokens: {input_ids}")

        for i in range(length):
            context = generated[-block_size:]
            context = [0] * (block_size - len(context)) + context if len(context) < block_size else context

            logits = model(tf.constant([context], dtype=tf.int32), training=False)
            next_id = tf.random.categorical(logits[0, -1:] / temperature, 1).numpy()[0][0]
            generated.append(next_id)

            # Stop if we hit end token (commonly 0 or specific end token)
            if next_id == 0:
                break

        # Try different decoding approaches
        try:
            return tokenizer.sequences_to_texts([generated])[0]
        except Exception:
            # Fallback: manual decoding if sequences_to_texts fails
            return ' '.join([tokenizer.index_word.get(i, '<UNK>') for i in generated if i > 0])

    except Exception as e:
        return f"Error generating text: {str(e)}"


# === 7. Try Text Generation ===
print('Ask everything')

while True:
    prompt = input('> ')

    if prompt == 'q!':
        print('Bye!')
        exit(0)

    response = generate_text(prompt)
    print(response)
