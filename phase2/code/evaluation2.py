import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


import random
import os

import entropy as ent
import math
from sklearn.model_selection import KFold

from sklearn import tree
from sklearn.cluster import KMeans




seed = 57
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


def DT(f_x, f_y):

    x_train, x_test, y_train, y_test = train_test_split(f_x,f_y,random_state=seed,test_size=0.2)

    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)

    svm_scores = cross_val_score(clf, f_x, np.ravel(f_y), cv = 5)
    # print("Cross Validation Score {}".format(svm_scores.mean()))

    return accuracy


def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]

def ClusterIndicesComp(clustNum, labels_array): #list comprehension
    return np.array([i for i, x in enumerate(labels_array) if x == clustNum])


def KMEANS(X):
    kmeans = KMeans(n_clusters=3, n_init=3)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    return kmeans



all_X = pickle.load(open('tex.pkl', 'rb'))


_1 = np.full((200,1), 1)
_2 = np.full((100,1), 2)
_3 = np.full((100,1), 3)

# AB vs E
X1 = all_X[200:]
Y1 = np.concatenate((_1, _2), axis=0)

# CD vs E
X2 = np.concatenate((all_X[:200], all_X[400:]), axis=0) 
Y2 = np.concatenate((_1, _2), axis=0)

# D vs R
X3 = np.concatenate((all_X[:100], all_X[400:]), axis=0)
Y3 = np.concatenate((_2, _3), axis=0)


def AB_vs_E(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=seed,test_size=0.2)

    km = KMEANS(x_train)

    cluster1 = ClusterIndicesNumpy(0, km.labels_)
    cluster2 = ClusterIndicesNumpy(1, km.labels_)
    cluster3 = ClusterIndicesNumpy(2, km.labels_)


    cluster1_x = x_train[cluster1]
    cluster2_x = x_train[cluster2]
    cluster3_x = x_train[cluster3]


    cluster1_y = y_train[cluster1]
    cluster2_y = y_train[cluster2]
    cluster3_y = y_train[cluster3]


    clf1 = tree.DecisionTreeClassifier()
    clf1.fit(cluster1_x, cluster1_y)

    clf2 = tree.DecisionTreeClassifier()
    clf2.fit(cluster2_x, cluster2_y)

    clf3 = tree.DecisionTreeClassifier()
    clf3.fit(cluster3_x, cluster3_y)


    score = 0
    for (x_data, y_data) in zip(x_test, y_test):
        closest_cluster = km.predict(x_data.reshape(1,8))
        if closest_cluster == 0:
            y_pred = clf1.predict(x_data.reshape(1,8))
            print("predicted {} - value {}".format(y_pred, y_data))
            if (y_pred == y_data):
                score += 1
        elif closest_cluster == 1:
            y_pred = clf2.predict(x_data.reshape(1,8))
            print("predicted {} - value {}".format(y_pred, y_data))
            if (y_pred == y_data):
                score += 1
        else:
            y_pred = clf3.predict(x_data.reshape(1,8))
            print("predicted {} - value {}".format(y_pred, y_data))
            if (y_pred == y_data):
                score += 1

    
    return score


def CD_vs_E(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=seed,test_size=0.2)

    km = KMEANS(x_train)

    cluster1 = ClusterIndicesNumpy(0, km.labels_)
    cluster2 = ClusterIndicesNumpy(1, km.labels_)
    cluster3 = ClusterIndicesNumpy(2, km.labels_)


    cluster1_x = x_train[cluster1]
    cluster2_x = x_train[cluster2]
    cluster3_x = x_train[cluster3]


    cluster1_y = y_train[cluster1]
    cluster2_y = y_train[cluster2]
    cluster3_y = y_train[cluster3]


    clf1 = tree.DecisionTreeClassifier()
    clf1.fit(cluster1_x, cluster1_y)

    clf2 = tree.DecisionTreeClassifier()
    clf2.fit(cluster2_x, cluster2_y)

    clf3 = tree.DecisionTreeClassifier()
    clf3.fit(cluster3_x, cluster3_y)


    score = 0
    for (x_data, y_data) in zip(x_test, y_test):
        closest_cluster = km.predict(x_data.reshape(1,8))
        if closest_cluster == 0:
            y_pred = clf1.predict(x_data.reshape(1,8))
            print("predicted {} - value {}".format(y_pred, y_data))
            if (y_pred == y_data):
                score += 1
        elif closest_cluster == 1:
            y_pred = clf2.predict(x_data.reshape(1,8))
            print("predicted {} - value {}".format(y_pred, y_data))
            if (y_pred == y_data):
                score += 1
        else:
            y_pred = clf3.predict(x_data.reshape(1,8))
            print("predicted {} - value {}".format(y_pred, y_data))
            if (y_pred == y_data):
                score += 1

    
    return score


def D_vs_E(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=seed,test_size=0.2)

    km = KMEANS(x_train)

    cluster1 = ClusterIndicesNumpy(0, km.labels_)
    cluster2 = ClusterIndicesNumpy(1, km.labels_)
    cluster3 = ClusterIndicesNumpy(2, km.labels_)


    cluster1_x = x_train[cluster1]
    cluster2_x = x_train[cluster2]
    cluster3_x = x_train[cluster3]


    cluster1_y = y_train[cluster1]
    cluster2_y = y_train[cluster2]
    cluster3_y = y_train[cluster3]


    clf1 = tree.DecisionTreeClassifier()
    clf1.fit(cluster1_x, cluster1_y)

    clf2 = tree.DecisionTreeClassifier()
    clf2.fit(cluster2_x, cluster2_y)

    clf3 = tree.DecisionTreeClassifier()
    clf3.fit(cluster3_x, cluster3_y)


    score = 0
    for (x_data, y_data) in zip(x_test, y_test):
        closest_cluster = km.predict(x_data.reshape(1,8))
        if closest_cluster == 0:
            y_pred = clf1.predict(x_data.reshape(1,8))
            print("predicted {} - value {}".format(y_pred, y_data))
            if (y_pred == y_data):
                score += 1
        elif closest_cluster == 1:
            y_pred = clf2.predict(x_data.reshape(1,8))
            print("predicted {} - value {}".format(y_pred, y_data))
            if (y_pred == y_data):
                score += 1
        else:
            y_pred = clf3.predict(x_data.reshape(1,8))
            print("predicted {} - value {}".format(y_pred, y_data))
            if (y_pred == y_data):
                score += 1

    
    return score


# score = AB_vs_E(X1, Y1)
# print(score)

# score = CD_vs_E(X2, Y2)
# print(score)

# score = D_vs_E(X3, Y3)
# print(score)