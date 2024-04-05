import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import tensorflow.data as tf_data

import keras_nlp
import keras

DIR = '../assets/gpt/'
FILENAME_TRAIN_DS = 'dialogues_train.txt'
FILENAME_VALID_DS = 'dialogues_validation.txt'

# Data
BATCH_SIZE = 16
SEQ_LEN = 32

# Model
EMBED_DIM = 256
FEED_FORWARD_DIM = 128
NUM_HEADS = 3
NUM_LAYERS = 3
VOCAB_SIZE = 5000

raw_train_ds = (
    tf_data.TextLineDataset(DIR + FILENAME_TRAIN_DS)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=2048, reshuffle_each_iteration=True)
)

raw_val_ds = (
    tf_data.TextLineDataset(DIR + FILENAME_VALID_DS)
    .batch(BATCH_SIZE)
)

vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN
)

start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id('[BOS]'),
)


def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels


train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(
    tf_data.AUTOTUNE
)

val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(
    tf_data.AUTOTUNE
)

inputs = keras.layers.Input(shape=(None,), dtype='int32')

embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)
x = embedding_layer(inputs)

for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM,
    )
    x = decoder_layer(x)

outputs = keras.layers.Dense(VOCAB_SIZE)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)

model.compile(optimizer='adam', loss=loss_fn, metrics=[perplexity])
model.fit(train_ds, validation_data=val_ds, epochs=10)

while True:

    prompt = input('Your input: ')

    if prompt == 'exit':
        print('Exiting...')
        break

    if prompt == '':
        continue

    prompt_tokens = start_packer(tokenizer([prompt]))

    def next_token(prmt, cache, index):
        logits = model(prmt)[:, index - 1, :]
        hidden_states = None
        return logits, hidden_states, cache


    sampler = keras_nlp.samplers.TopKSampler(k=10)

    output_tokens = sampler(
        next=next_token,
        prompt=prompt_tokens,
        index=1
    )

    txt = tokenizer.detokenize(output_tokens)
    print(f"{txt}\n")
