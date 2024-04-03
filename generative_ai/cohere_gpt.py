import sys

import cohere
from cohere import ClassifyExample

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
np.set_printoptions(threshold=sys.maxsize, suppress=True)

co = cohere.Client('PjvfZ9sPIgqHTUhfMOTJVRCG6aWbpb48w7zpihdV')


def collaborate():
    print('Initial interaction with the Cohere model')

    while True:
        prompt = input('>>> ')

        if prompt == 'exit':
            break

        if prompt == '':
            continue

        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=200,
            temperature=1)

        output = response.generations[0].text
        print(output)


def classify_sentiment():
    examples = [
         ClassifyExample(text="I'm so proud of you", label='positive'),
         ClassifyExample(text="What a great time to be alive", label='positive'),
         ClassifyExample(text="That's awesome work", label='positive'),
         ClassifyExample(text="The service was amazing", label='positive'),
         ClassifyExample(text="I love my family", label='positive'),
         ClassifyExample(text="They don't care about me", label='negative'),
         ClassifyExample(text="I hate this place", label='negative'),
         ClassifyExample(text="The most ridiculous thing I've ever heard", label='negative'),
         ClassifyExample(text="I am really frustrated", label='negative'),
         ClassifyExample(text="This is so unfair", label='negative'),
         ClassifyExample(text="This made me think", label='neutral'),
         ClassifyExample(text="The good old days", label='neutral'),
         ClassifyExample(text="What's the difference", label='neutral'),
         ClassifyExample(text="You can't ignore this", label='neutral'),
         ClassifyExample(text="That's how I see it", label='neutral')
         ]

    inputs = [
        "Hello, world! What a beautiful day",
        "This is the worst thing",
        "That's how I see it",
        'My wife is angry!!'
    ]

    results = co.classify(
        model='embed-english-v2.0',
        examples=examples,
        inputs=inputs,

    )

    for r in results.classifications:
        print(
            'Input: {} | Prediction: {} | Confidence: {}'.format(r.input, r.prediction, r.confidence)
        )


def semantic_search_exploration(plotting=False):
    df = pd.read_csv('../assets/gpt/hello-world-kw.csv', names=['search_term'])

    def embed_text(input_type, texts):
        response = co.embed(
            model='embed-english-v3.0',
            input_type=input_type,
            texts=texts
        )

        return response.embeddings

    def get_similarity(target, candidates):

        # Turning list into array
        candidates = np.array(candidates)
        target = np.expand_dims(np.array(target), axis=0)

        # Calculate cosine similarity
        sim = cosine_similarity(target, candidates)
        sim = np.squeeze(sim).tolist()

        # Sort by descending order in similarity
        sim = list(enumerate(sim))
        sim = sorted(sim, key=lambda x: x[1], reverse=True)

        return sim

    df['search_term_embeds'] = embed_text(texts=df['search_term'].tolist(), input_type='search_document')
    doc_embeds = np.array(df['search_term_embeds']).tolist()

    query = 'what is the history of hello world'
    query_embeds = embed_text(texts=[query], input_type='search_query')[0]

    print('\nQuery: ', query)

    similarity = get_similarity(query_embeds, doc_embeds)
    for idx, score in similarity[:5]:
        print(f'Similarity: {score:.2f};', df.iloc[idx]['search_term'])

    # Semantic exploration
    if plotting is True:
        reducer = PCA(n_components=2)
        red_embeds = reducer.fit_transform(doc_embeds)

        fig, ax = plt.subplots()
        ax.scatter(red_embeds[:, 0], red_embeds[:, 1])
        ax.set_title('Compressed embeddings on a 2D plot')

        for i, txt in enumerate(df['search_term']):
            ax.annotate(txt, (red_embeds[i, 0], red_embeds[i, 1]))

        plt.show()


if __name__ == '__main__':
    # collaborate()
    classify_sentiment()
    semantic_search_exploration(plotting=True)
