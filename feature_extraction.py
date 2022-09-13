import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.spatial.distance import euclidean

import numpy as np
import pandas as pd

from scipy.stats import beta, shapiro
from scipy.stats import lognorm

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro

texts = ["i have a cat", "you have a dog", "you and i have a cat and a dog"]

vocabulary = list(enumerate(set([word for sentence in texts for word in sentence.split()])))
# print('Vocalbulary', list(vocabulary))


def vectorize(text):
    vector = np.zeros(len(vocabulary))
    for i, word in vocabulary:
        num = 0
        for w in text:
            if w == word:
                num += 1

        vector[i] = num

    return vector


def extract_text():
    v1 = vectorize(texts[0].split())
    v2 = vectorize(texts[1].split())
    v3 = vectorize(texts[2].split())
    print(v1, v2, v3)
    print(euclidean(v1, v2), euclidean(v2, v3))

    vect = CountVectorizer(ngram_range=(1, 1))
    v1, v2, v3 = vect.fit_transform(texts).toarray()
    print(v1, v2, v3)
    print(euclidean(v1, v2), euclidean(v2, v3))

    print(vect.vocabulary_)


def is_data_normally_distributed():
    x = np.arange(1, 10 + 1)

    # data = np.random.rand(1, 10, 100)
    y = lognorm(s=1).rvs(10)

    # scaler = MinMaxScaler()
    # scaled_data = scaler.fit_transform(data.reshape(1, -1))

    print(x)

    plt.plot(x, y)
    plt.show()


def plot_qq():
    df = pd.read_csv('assets/petrol_consumption.csv')

    consumption = df['Petrol_Consumption'].values
    consumption_log = np.log(consumption)

    scaled_min_max = MinMaxScaler().fit_transform(consumption.reshape(-1, 1)).flatten()
    scaled_std = StandardScaler().fit_transform(consumption.reshape(-1, 1)).flatten()

    # sm.qqplot(consumption, loc=consumption.mean(), scale=consumption.std())
    # sm.qqplot(scaled_min_max, loc=scaled_min_max.mean(), scale=scaled_min_max.std())
    sm.qqplot(scaled_std, loc=scaled_std.mean(), scale=scaled_std.std())
    sm.qqplot(consumption_log, loc=consumption_log.mean(), scale=consumption_log.std())

    plt.show()


def select_features_variance_threshold():
    df = pd.read_csv('assets/breast_dataset.csv')

    X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
    y = df['diagnosis']

    print(X.shape)

    # Remove features with low variance
    selector = VarianceThreshold(0.7)
    X_shaped = selector.fit_transform(X)

    print(X_shaped.shape)


# is_data_normally_distributed()
# plot_qq()

