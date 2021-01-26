'''
Based on https://towardsdatascience.com/how-to-build-knn-from-scratch-in-python-5e22b8920bd2
'''

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

#loading and setting up the iris data set
iris = datasets.load_iris()

df = pd.DataFrame(data=iris.data, columns =iris.feature_names)
df['target'] = iris.target
df.head()

X = df.drop('target',axis=1)
y = df.target

#split a third of the training data into a validation set. Seed of 0
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#calculate distance between two data samples
def euclidean_distance(sampleA,sampleB):
    dim = len(sampleA)

    distance = 0

    for index in range(dim):
        toAdd = (sampleA[index] - sampleB[index]) * (sampleA[index] - sampleB[index])
        distance = distance + toAdd

    distance = math.sqrt(distance)
    return distance

test_pt = [4.8, 2.7, 2.5, 0.7]

#for given flower data, return the predicted label
def getLabel(data, n_neighbors):
    distances = []

    for i in X.index:
        distances.append(euclidean_distance(data, X.iloc[i]))

    df_dists = pd.DataFrame(data=distances, index=X.index, columns=['dist'])
    df_dists.head()

    df_nn = df_dists.sort_values(by=['dist'], axis=0)[:n_neighbors]

    counter = Counter(y[df_nn.index])

    label = counter.most_common()[0][0]
    return label

y_predict = []
for flower in X_test.values:
    y_predict.append(getLabel(flower,15))

score = accuracy_score(y_test,y_predict)