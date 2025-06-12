import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import json
import pickle

import tensorflow as tf
import numpy as np

from llm_dataset import LlmTokenizer
from llm_model import fast_label_smoothing_loss

from llm_dataset import (
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
model_id = 'llm_model_114_3000'
model_pathname = f"{MODELS_DIRECTORY_PATH}/{model_id}"

model = tf.keras.models.load_model(
    model_pathname,
    custom_objects={
        'fast_label_smoothing_loss': fast_label_smoothing_loss
    })

print(f"Model {model_id} has been loaded")

# Load the model's metadata
with open(f"{MODELS_DIRECTORY_PATH}/{model_id}/{MODEL_METADATA_FILENAME}", 'r') as f:
    m_metadata = json.load(f)

    print(f"The Model's metadata has been loaded\n{m_metadata}\n")

    # Reading the required model's hyperparameters
    block_size = m_metadata.get('block_size')


def generate_text(prompt, length=20, temperature=1.0, block_size=block_size):
    input_ids = tokenizer.texts_to_sequences([prompt])[0]
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
        logits = model(input_tensor, training=False)

        # Use sampling or deterministic argmax decoding
        next_token = tf.random.categorical(logits[0, -1:] / temperature, 1).numpy()[0][0]

        generated.append(next_token)
        context.append(next_token)

    # Decode generated tokens to text
    try:
        return tokenizer.sequences_to_texts([generated])[0]
    except Exception:
        return ' '.join([tokenizer.index_word.get(i, '<UNK>') for i in generated if i > 0])


# === 7. Try Text Generation ===
print('Ask everything')

while True:
    prompt = input('> ')

    if prompt == 'q!':
        print('Bye!')
        exit(0)

    response = generate_text(prompt)
    print(response)
