# Guideline: https://365datascience.com/tutorials/how-to-build-recommendation-system-in-python/

import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)
np.set_printoptions(threshold=sys.maxsize, suppress=True)


colnames = [
    'ISBN',
    'Book-Title',
    'Book-Author',
    'Year-Of-Publication',
    'Publisher',
    'Image-URL-S',
    'Image-URL-M',
    'Image-URL-L'
]
df = pd.read_csv('../assets/books.csv', names=colnames)

# Removing duplicates
df = df.drop_duplicates(subset='Book-Title')

# Picking the required columns
df = df[['Book-Title', 'Book-Author', 'Publisher']]


def append_book_author(author):
    """
    Removes whitespaces from the “Book-Author” column and appends the author's name to a single word
    :param author
    :return: appended author's name
    """

    return str(author).lower().replace(' ', '')


df['Book-Author'] = df['Book-Author'].apply(append_book_author)
df['Book-Title'] = df['Book-Title'].str.lower()
df['Publisher'] = df['Publisher'].str.lower()

# Combining the dataset's columns to a single column (variable)
df['data'] = df[df.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

vect = CountVectorizer()
vect_data = vect.fit_transform(df['data'])

# Cosine similarity ranges between 0 and 1.
# A value of 0 indicates that the two vectors are not similar at all, while 1 tells us that they are identical.
# https://en.wikipedia.org/wiki/Cosine_similarity
similarities = cosine_similarity(vect_data)

# Decode the book's title, mapping them to the similar books
df = pd.DataFrame(similarities, columns=df['Book-Title'], index=df['Book-Title']).reset_index().round(3)

# Recommending
input_book = 'the angel is near'

recommendations = pd.DataFrame(df.nlargest(11, input_book)['Book-Title'])
recommendations = recommendations[recommendations['Book-Title'] != input_book]

print(recommendations)


def count_vectorizer_sample():
    """
    CountVectorizer converts a collection of documents into a vector of word counts.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """

    text = [
        'Anne and James both like to play video games and football.',
        'Anne likes video games more than James does.'
    ]

    vect = CountVectorizer()
    X = vect.fit_transform(text)

    print(vect.get_feature_names_out())
    print(X.toarray())

# count_vectorizer_sample()
