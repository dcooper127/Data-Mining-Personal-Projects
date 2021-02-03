from sklearn import datasets
from sklearn import decomposition
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

iris = datasets.load_iris()

X = iris.data

#not shown: using PCA to view explained variance. ~roughly 97% of variance was explained by first 2 compononets, so third one was dropped
pca = decomposition.PCA(n_components=2)
pca.fit(X)


X = pca.transform(X)

#
X = pd.DataFrame(data=X)
X['target'] = iris.target


y = X.target

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

    for dat in X_train.values:
        distances.append(euclidean_distance(data,dat))

    df_dists = pd.DataFrame(data=distances, index=X_train.index, columns=['dist'])
    df_dists.head()

    df_nn = df_dists.sort_values(by=['dist'], axis=0)[:n_neighbors]

    counter = Counter(y[df_nn.index])

    label = counter.most_common()[0][0]
    return label

    return distances

y_predict = []
for flower in X_test.values:
    y_predict.append(getLabel(flower,15))

score = accuracy_score(y_test,y_predict)