# Guideline:
# https://keras.io/examples/generative/text_generation_gpt/
# https://www.kdnuggets.com/2018/05/wtf-tensor.html


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

import tensorflow.data as tf_data
import tensorflow as tf

import keras
import keras_nlp

# The original dataset is Simple Books, can be downloaded at
# https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip

# Files data
DIR = '../assets/gpt/'
FILENAME_TRAIN_DS = 'simple_books_short_train.txt'
FILENAME_VALID_DS = 'simple_books_short_valid.txt'

# Data
BATCH_SIZE = 64
MIN_STRING_LEN = 512  # Strings shorter than this will be discarded
SEQ_LEN = 128  # Length of training sequences, in tokens

# Model
EMBED_DIM = 256
FEED_FORWARD_DIM = 128
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000  # Limits parameters in model.

# Training
EPOCHS = 1

# LOAD TRAIN AND VALIDATION DATASETS

# The process of training LLMs involves feeding them a large amount of text data, known as the training set.
# This data is used to adjust the model’s parameters so that it can accurately predict the next word in a sentence.
# The model learns the statistical patterns in the data, which it then uses to generate text.

# The tf.data.TextLineDataset loads text from text files and creates a dataset where each line of the files becomes
# an element of the dataset.
raw_train_ds = (
    tf_data.TextLineDataset(DIR + FILENAME_TRAIN_DS)
    .filter(lambda x: tf.strings.length(x) > MIN_STRING_LEN)

    # A tf.int64 scalar tf.Tensor, representing the number of consecutive elements
    # of this dataset to combine in a single batch.
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

# https://www.chatgptguide.ai/2024/03/03/what-is-validation-set-llms-explained/
# A validation file is examples of the same quality as your training that are held out. At the end of batches during
# fine-tuning, the generation is run on the trained model vs the validation examples to find the deviation by token
# scoring. One can chart the progress to find where the AI model has become optimized for the full breadth of
# questions, while not becoming over-trained on specifically the inputs and outputs that are just in the training set.
raw_val_ds = (
    tf_data.TextLineDataset(DIR + FILENAME_VALID_DS)
    .filter(lambda x: tf.strings.length(x) > MIN_STRING_LEN)
    .batch(BATCH_SIZE)
)


# TRAIN TOKENIZER VOCABULARY
# https://huggingface.co/learn/nlp-course/en/chapter6/6
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)

# https://keras.io/api/keras_nlp/tokenizers/word_piece_tokenizer/
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN
)

# Packer adds a start token
# https://keras.io/api/keras_nlp/preprocessing_layers/start_end_packer/
start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id('[BOS]'),
)


def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels


# Tokenize and split into train and label sequences
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch
# https://www.tensorflow.org/api_docs/python/tf/raw_ops/PrefetchDataset
train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(
    tf_data.AUTOTUNE
)

val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(
    tf_data.AUTOTUNE
)


# BUILDING THE NN MODEL
inputs = keras.layers.Input(shape=(None,), dtype='int32')

# Embedding
# Token and position embeddings are ways of representing words and their order in a sentence.
# https://keras.io/api/keras_nlp/modeling_layers/token_and_position_embedding/
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)
x = embedding_layer(inputs)


# Transformer decoders
# https://keras.io/api/keras_nlp/modeling_layers/transformer_decoder/
for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM,
    )
    x = decoder_layer(x)  # Giving one argument only skips cross-attention.

# Output
outputs = keras.layers.Dense(VOCAB_SIZE)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# “logits” refer to the raw, unnormalized predictions generated by the last layer of a neural network
# before applying an activation function. Essentially, logits are the scores assigned to each class or category,
# reflecting the model’s confidence in its predictions.
# https://keras.io/api/losses/probabilistic_losses/#sparsecategoricalcrossentropy-class
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Perplexity in language models is like a game of guessing the next word in a sentence;
# the better the model is at guessing, the lower the perplexity score.
# https://klu.ai/glossary/perplexity
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)

model.compile(optimizer='adam', loss=loss_fn, metrics=[perplexity])

model.summary()

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# INFERENCE
while True:

    prompt = input('Your input: ')

    if prompt == 'exit':
        print('Exiting...')
        break

    prompt_tokens = start_packer(tokenizer([prompt]))

    # The wrapper function
    def next_token(prmt, cache, index):
        logits = model(prmt)[:, index - 1, :]
        # Ignore hidden states for now; only needed for contrastive search.
        hidden_states = None
        return logits, hidden_states, cache

    # Different samplers

    # Greedily picks the most probable token at each timestep.
    # In other words, we get the argmax of the model output.
    # sampler = keras_nlp.samplers.GreedySampler()

    # At a high-level, Beam search keeps track of the num_beams most probable sequences at each timestep,
    # and predicts the best next token from all sequences
    # sampler = keras_nlp.samplers.BeamSampler(num_beams=10)

    # At each time step, Random Sampler samples the next token using the softmax probabilities
    # provided by the model.
    # sampler = keras_nlp.samplers.RandomSampler()

    # Similar to random search, we sample the next token from the probability distribution provided by the model.
    # The only difference is that here, TopKSampler selects out the top k most probable tokens, and distribute the
    # probability mass over them before sampling.
    # sampler = keras_nlp.samplers.TopKSampler(k=10)

    #  Instead of choosing a k, we choose a probability p that we want the probabilities of the top tokens
    #  to sum up to. This way, we can dynamically adjust the k based on the probability distribution.
    #  By setting p=0.9, if 90% of the probability mass is concentrated on the top 2 tokens, we can
    #  filter out the top 2 tokens to sample from.
    sampler = keras_nlp.samplers.TopPSampler(p=0.5)

    output_tokens = sampler(
        next=next_token,
        prompt=prompt_tokens,
        index=1,  # Start sampling immediately after the [BOS] token.
    )

    txt = tokenizer.detokenize(output_tokens)
    print(f"{txt}\n")