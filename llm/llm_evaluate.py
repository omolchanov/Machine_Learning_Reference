import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import math
import time
import pickle

import tensorflow as tf
import numpy as np

import nltk
nltk.download('punkt', quiet=True)
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from datasets import load_dataset

from llm_model import (
    MODELS_DIRECTORY_PATH,
    MODEL_METADATA_FILENAME
)

from llm_tokenizer import (
    DATA_DIRECTORY_PATH,
    DS_FIlENAME,
    TOKENIZER_FILENAME
)

MODEL_EVAL_DATA_FILENAME = 'model_evaluation.json'


# Load tokenizer
with open(f"{DATA_DIRECTORY_PATH}/{TOKENIZER_FILENAME}", 'rb') as f:
    tokenizer = pickle.load(f)

# Load dataset
data = np.load(f"{DATA_DIRECTORY_PATH}/{DS_FIlENAME}")

# Load the model
model_id = 'llm_model_702_0.069_200'
model_pathname = f"{MODELS_DIRECTORY_PATH}/{model_id}"

model = tf.keras.models.load_model(model_pathname)
print(f"Model {model_id} has been loaded")

# Load the model's metadata
with open(f"{model_pathname}/{MODEL_METADATA_FILENAME}", 'r') as f:
    m_metadata = json.load(f)
    print(f"The Model's metadata has been loaded\n{m_metadata}\n")

    # Setting the required model's metadata
    loss = m_metadata.get('loss')
    block_size = m_metadata.get('block_size')

# # Load ARC dataset (replace with actual dataset path)
# dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
# arc_data = dataset["train"].to_list()
#
#
# correct = 0
# total = len(arc_data)
#
# # Evaluate ARC accuracy
# for item in arc_data:
#     question = item["question"]
#     choices = item["choices"]
#     correct_answer = item["choices"][item["answerKey"]]
#
#     # Tokenize input question
#     input_tokens = tokenizer.encode(question)[:block_size]  # Ensure input fits model's block size
#     input_padded = np.pad(input_tokens, (0, block_size - len(input_tokens)), mode='constant')
#
#     # Get model prediction
#     logits = model.predict(np.array([input_padded]), verbose=0)
#     predicted_token = np.argmax(logits[0, -1])  # Get the highest probability token
#     predicted_answer = tokenizer.decode([predicted_token])  # Convert back to text
#
#     if predicted_answer == correct_answer:
#         correct += 1
#
# accuracy = correct / total
# print(f"ARC Accuracy: {accuracy:.2%}")


"""
=== Perplexity Evaluation ===

Perplexity is a metric that measures the uncertainty of a model's predictions. Specifically, in language models, 
it quantifies how well the model predicts the next word in a sequence. When a model makes a prediction, 
it assigns probabilities to possible next words.
Perplexity measures how well the LLM predicts a sequence of words. 
A lower perplexity indicates that the model is more confident in its predictions because it assigns higher probabilities 
to the actual sequence of words.
"""
perplexity = math.exp(loss)
print(f"Perplexity: {perplexity:.3f}")


"""
=== Log-likelihood Evaluation ===

The log-likelihood value of a regression model is a way to measure the goodness of fit for a model. 
The higher the value of the log-likelihood, the better a model fits a dataset.
The log-likelihood value for a given model can range from negative infinity to positive infinity. 
The actual log-likelihood value for a given model is mostly meaningless, but it’s useful for comparing two or 
more models.

In practice, we often fit several regression models to a dataset and choose the model with the highest log-likelihood 
value as the model that fits the data best.

Log-likelihood is negative of cross-entropy loss
"""
avg_log_likelihood = -loss
print(f"Average log-likelihood per token: {avg_log_likelihood}")




def generate_text(model, seed_tokens, max_length=20):
    generated = list(seed_tokens)
    for _ in range(max_length):
        current_seq = generated[-block_size:]
        if len(current_seq) < block_size:
            current_seq = [0] * (block_size - len(current_seq)) + current_seq
        input_seq = np.array(current_seq)[None, :]
        logits = model.predict(input_seq, verbose=0)
        next_token = int(np.argmax(logits[0, -1]))
        generated.append(next_token)
    return generated


"""
=== Response time evaluation ===

Time required for generating a response from the model. The lower response time is a sign of a better model's 
performance
"""
seed_tokens = data[:block_size]

start_time = time.time()
generated_tokens = generate_text(model, list(seed_tokens), max_length=20)

gen_latency = time.time() - start_time
print(f"Generation latency: , {gen_latency:.3f}")


"""
=== Diversity evaluation ===

Diversity evaluation of Large Language Models (LLMs) assesses the ability of an LLM to produce a variety of outputs, 
rather than simply repeating the same patterns or responses. It's a crucial aspect of evaluating LLMs because it 
ensures they are not just generating fluent text, but also that their outputs are diverse and creative. 

Distinct-1 is a common metric for measuring the lexical diversity of generated text. 
It is calculated as the number of unique unigrams (single tokens) divided by the total number of unigrams 
in the generated output.

- High Distinct-1 Score: A value closer to 1 indicates that nearly every token in the text is unique, 
    suggesting that the model is generating a wide variety of words with little repetition. 
    This is generally desirable when diversity is a priority.

- Low Distinct-1 Score: A value closer to 0 indicates a high level of repetition, meaning the model is using 
    the same words over and over. This can be a sign of a less creative or overly conservative generation process.

Keep in mind that while a high Distinct-1 score signals diversity, it doesn't alone ensure that the text is coherent 
or contextually appropriate
"""


def distinct_n_tokens(token_list, n=1):
    ngrams = [tuple(token_list[i:i + n]) for i in range(len(token_list) - n + 1)]
    return len(set(ngrams)) / (len(ngrams) if ngrams else 1)


distinct_score = distinct_n_tokens(generated_tokens, n=1)
print(f"Distinct-1 score: {distinct_score:.3f}")


def tokens_to_text(token_list):
    return " ".join(map(str, token_list))


ref_tokens = list(data[block_size:block_size + 20])
reference_text = tokens_to_text(ref_tokens)
generated_text = tokens_to_text(generated_tokens)

"""
=== BLEU Score evaluation ===

BLEU (Bilingual Evaluation Understudy) is an automated metric commonly used to assess the quality of machine-generated 
text, particularly in translation tasks. It evaluates how closely a candidate sentence matches reference sentences by 
calculating n-gram overlaps.

Key Points:
- Range: BLEU scores range from 0 to 1, with 1 representing a perfect match with reference texts. 
    However, in practice, strong models usually score well below 1 (e.g., around 0.3–0.5).

- n-Gram Matching: It focuses on n-gram (e.g., unigram to 4-gram) overlaps. Higher-order n-grams (trigrams, 4-grams) 
    can be very strict. If a candidate sentence is short or phrased differently, it may yield a low score even when
    the translation is acceptable.

Interpretation:
- High BLEU: Indicates that the generated text closely follows the reference, meaning there's significant n-gram 
    overlap.
- Low BLEU: Suggests that while the candidate might be grammatically correct or contextually relevant, it uses different 
    wording or structures compared to the references.

In summary, while BLEU is useful for approximate quality checks, it works best when used alongside other evaluation 
metrics, as it may not capture semantic correctness or stylistic variations fully.
"""

bleu = sentence_bleu([reference_text.split()], generated_text.split())
print(f"BLEU score: {bleu:.100f}")


"""
=== ROUGE Score evaluation ===

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap between a generated text 
(e.g., a summary or translation) and one or more reference texts. It does so by comparing n-grams 
(e.g., unigrams for ROUGE-1), as well as using sequence-based methods like the longest common subsequence (ROUGE-L). 

The key scores are:
- Precision: What fraction of the generated n-grams are also in the reference.
- Recall: What fraction of the reference n-grams are captured in the generated text.
- F-measure: The harmonic mean of precision and recall, balancing both aspects.

ROUGE-1 F-measure is a metric that balances both precision and recall when evaluating 
text similarity at the word (unigram) level. Here’s how to interpret it:

- High ROUGE-1 F-measure (closer to 1): This suggests the generated text effectively captures important words from the 
    reference while maintaining minimal redundancy. It means the overlap is strong without excessive irrelevant words.

- Low ROUGE-1 F-measure (closer to 0): Indicates weaker overlap—either the generated text misses key words (low recall) 
    or contains too many additional words that don’t appear in the reference (low precision).
"""

scorer_obj = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
rouge_score = scorer_obj.score(reference_text, generated_text)['rouge1'][2]
print(f"Rouge-1 F-measure: {rouge_score:.4f}")




# === Save Model Evaluation data ===

data = {
    "model": model_id,
    "perplexity": perplexity,
    "avg_log_likelihood": avg_log_likelihood,
    "response_time": gen_latency,
    "distinct_score": distinct_score,
    "blue_score": bleu,
    "rouge_score": rouge_score
}

metadata_pathname = f"{model_pathname}/{MODEL_EVAL_DATA_FILENAME}"
with open(metadata_pathname, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\nThe model evaluation data has been saved to {metadata_pathname}")
