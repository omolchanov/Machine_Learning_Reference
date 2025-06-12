import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import time

from datasets import load_dataset
import numpy as np

import nltk
nltk.download('punkt', quiet=True)
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from llm_dataset import LlmTokenizer
from llm_model_entity import LlmModelEntity
from llm_dataset import LlmDataset
from env_config import *


MODEL_ID = 'llm_model_321_30'

tokenizer = LlmTokenizer.load_obj()
data = LlmDataset.load_ds()
model = LlmModelEntity.load(MODEL_ID)
m_metadata = LlmModelEntity.load_metadata(MODEL_ID)

block_size = m_metadata.get('block_size')
loss = m_metadata.get('loss')


"""
=== TruthfulQA evaluation ===
TruthfulQA is a benchmark designed to measure how well language models produce truthful and factual answers to 
challenging questions that often induce false or misleading responses. It tests if the model avoids common 
misconceptions, falsehoods, and "hallucinations."
The dataset contains questions that are tricky or adversarial, often about myths, science, or controversial topics.
Answers are scored as truthful or non-truthful based on human annotation.

The main metric is accuracy on truthful answers. It measures the percentage of model-generated answers that are 
considered factually correct or honest. A higher score means the model is better at avoiding hallucinations and 
providing accurate, trustworthy information: 

- Low (<40%)	Model frequently hallucinates or produces false info.
- Medium (~40-60%)	Model sometimes truthful but inconsistent.
- High (>60%)	Model reliably provides truthful answers.
"""
dataset = load_dataset("truthful_qa", "multiple_choice", split=f'validation[:{N_QUESTIONS}]')

correct = 0
total = len(dataset)

for item in dataset:
    question = item["question"]
    choices = item["mc1_targets"]["choices"]  # list of candidate answers
    labels = item["mc1_targets"]["labels"]  # list of 0/1 (1 = correct)
    label = item["mc1_targets"]["labels"].index(1)

    candidates = []
    for choice in choices:
        text = question + " " + choice
        tokens = tokenizer.encode(text)[:block_size]
        tokens += [0] * (block_size - len(tokens))
        candidates.append(tokens)

    inputs = np.array(candidates)
    logits = model.predict(inputs, verbose=0)

    # Score each candidate by max logit of last token
    scores = [log[-1].max() for log in logits]
    predicted_idx = np.argmax(scores)
    predicted_answer = choices[predicted_idx]

    if predicted_answer == label:
        correct += 1

tqa_score = correct / total
print(f"TruthfulQA Accuracy: {tqa_score:.2%}")


"""
=== MMLU evaluation ===

MMLU Metric (Massive Multitask Language Understanding)
What it measures: Accuracy on a wide range of multiple-choice questions from 57 subjects, 
testing broad knowledge and reasoning.

Metric: Simple accuracy — percentage of correctly answered questions.
- Below 30%: Poor — model struggles with general knowledge and reasoning.
- 30%–50%: Moderate — some knowledge but many gaps.
- 50%–70%: Good — solid general understanding, comparable to strong baseline models.
- Above 70%: Excellent — approaching expert-level or human performance on many subjects.
- Above 80–90%: State-of-the-art / near-human performance.
"""
dataset = load_dataset("openai/MMMLU", split=f'test[:{N_QUESTIONS}]')

correct = 0
total = len(dataset)

for item in dataset:
    question = item["Question"]
    choices = [item["A"], item["B"], item["C"], item["D"]]
    label = ["A", "B", "C", "D"].index(item["Answer"])

    candidates = []
    for choice in choices:
        text = question + " " + choice
        tokens = tokenizer.encode(text)[:block_size]
        tokens += [0] * (block_size - len(tokens))
        candidates.append(tokens)

    inputs = np.array(candidates)
    logits = model.predict(inputs, verbose=0)
    scores = [log[-1].max() for log in logits]
    predicted = np.argmax(scores)

    if predicted == label:
        correct += 1

mmlu_score = correct / total
print(f"MMLU Accuracy: {mmlu_score:.2%}")


"""
=== HellaSwag evaluation ===

HellaSwag is a benchmark dataset designed to test commonsense reasoning and grounded natural language understanding. 
Models are given a context and four possible endings, and the task is to choose the most plausible continuation.

The primary metric for HellaSwag is accuracy. It measures the percentage of times your model correctly picks the one 
true (most plausible) ending out of the four options. Accuracy ranges from 0% (always wrong) to 100% (always right).

- High accuracy (~>80%) means your model is strong at commonsense reasoning and understanding real-world situations.
- Low accuracy (~<40%) indicates your model struggles with understanding context or reasoning beyond surface patterns.
- HellaSwag is intentionally challenging; human accuracy is usually around 90-95%.
"""
dataset = load_dataset("hellaswag", split=f'validation[:{N_QUESTIONS}]', trust_remote_code=True)

correct = 0
total = len(dataset)

for item in dataset:
    context = item["ctx"]
    choices = item["endings"]
    label = item["label"]

    candidates = [context + " " + ending for ending in choices]

    # Tokenize and pad each candidate
    inputs = []
    for text in candidates:
        tokens = tokenizer.encode(text)[:block_size]
        tokens += [0] * (block_size - len(tokens))
        inputs.append(tokens)

    inputs = np.array(inputs)

    # Predict and score
    logits = model.predict(inputs, verbose=0)
    scores = [log[-1].max() for log in logits]
    predicted = np.argmax(scores)

    if predicted == label:
        correct += 1

hellaswag_score = correct / total
print(f"HellaSwag Accuracy: {hellaswag_score:.2%}")


"""
=== ARC Evaluation ===

ARC stands for the AI2 Reasoning Challenge. It’s a benchmark designed to test reasoning and commonsense intelligence 
in AI systems. The ARC dataset consists of multiple-choice science questions (from grade school level science exams) 
with 4 answer choices. Each question comes with a context, question text, and answer options.

How to Interpret ARC Accuracy: 
- 25% (random)	Model is guessing at random (baseline for 4 choices).
- 40–50%	Somewhat better than chance — picking up patterns.
- 60–70%	Strong reasoning — good for open-domain models.
- 80%+	Very high performance, likely with fine-tuned LLMs.
"""
dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
arc_data = dataset["train"].to_list()

correct = 0
total = len(arc_data)

# Evaluate ARC accuracy
for item in arc_data:
    question = item["question"]
    choices = item["choices"]

    labels = item["choices"]["label"]
    texts = item["choices"]["text"]
    answer_key = item["answerKey"]
    correct_answer = texts[labels.index(answer_key)]

    # Tokenize input question
    input_tokens = tokenizer.encode(question)[:block_size]  # Ensure input fits model's block size
    input_padded = np.pad(input_tokens, (0, block_size - len(input_tokens)), mode='constant')

    # Get model prediction
    logits = model.predict(np.array([input_padded]), verbose=0)
    predicted_token = np.argmax(logits[0, -1])  # Get the highest probability token
    predicted_answer = tokenizer.decode([predicted_token])  # Convert back to text

    if predicted_answer == correct_answer:
        correct += 1

arc_score = correct / total
print(f"ARC Accuracy: {arc_score:.2%}")


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
print(f"Generation latency: {gen_latency:.3f}")


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
    "model": MODEL_ID,
    "arc_score": arc_score,
    "hellaswag_score": hellaswag_score,
    "mmlu_score": mmlu_score,
    "tqa_score": tqa_score,
    "perplexity": perplexity,
    "avg_log_likelihood": avg_log_likelihood,
    "response_time": gen_latency,
    "distinct_score": distinct_score,
    "blue_score": bleu,
    "rouge_score": rouge_score
}

LlmModelEntity.save_evaluation(MODEL_ID, data)
