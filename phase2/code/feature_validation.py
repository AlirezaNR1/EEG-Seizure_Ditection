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
import pandas as pd
import seaborn as sns


from sklearn.cluster import KMeans

seed = 57
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

x = pickle.load(open('ex.pkl', 'rb'))
y = pickle.load(open('ey.pkl', 'rb'))


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
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    return kmeans


accuracy_scores = {}
correlation_cal_data = {}
selected_features = []


#################### TIME DOMAIN FEATURES ####################

###### PTP ###### 
ptp = x[:, 0]
ptp_accuracy = DT(ptp.reshape(500,1), y)
accuracy_scores["PTP"] = ptp_accuracy
correlation_cal_data["PTP"] = ptp


###### AASS ######
aass = x[:, 1]
aass_accuracy = DT(aass.reshape(500,1), y)
accuracy_scores["AASS"] = aass_accuracy
correlation_cal_data["AASS"] = aass


###### SSA ######
ssa = x[:, 2]
ssa_accuracy = DT(ssa.reshape(500,1), y)
accuracy_scores["SSA"] = ssa_accuracy
correlation_cal_data["SSA"] = ssa


###### Log Detect ######
ld = x[:, 3]
ld_accuracy = DT(ld.reshape(500,1), y)
accuracy_scores["LD"] = ld_accuracy
correlation_cal_data["LD"] = ld


###### Zero Crossings(ZC) ###### 
zc = x[:, 4]
zc_accuracy = DT(zc.reshape(500,1),y)
accuracy_scores["ZC"] = zc_accuracy
correlation_cal_data["ZC"] = zc



#################### STATISTICAL FEATURES ####################

###### Mean ######
mean = x[:,5]
mean_accuracy = DT(mean.reshape(500,1),y)
accuracy_scores["Mean"] = mean_accuracy
correlation_cal_data["Mean"] = mean



###### Median ######
median = x[:,6]
median_accuracy = DT(median.reshape(500,1),y)
accuracy_scores["Median"] = median_accuracy
correlation_cal_data["Median"] = median



###### Percentile ######
percentile = x[:,7]
percentile_accuracy = DT(percentile.reshape(500,1),y)
accuracy_scores["Percentile"] = percentile_accuracy
correlation_cal_data["Percentile"] = percentile


###### Standard Derivation(STD) ###### 
std = x[:, 8]
std_accuracy = DT(std.reshape(500,1),y)
accuracy_scores["STD"] = std_accuracy
correlation_cal_data["STD"] = std



###### Histogram ######
histogram = x[:, 9]
histogram_accuracy = DT(histogram.reshape(500,1),y)
accuracy_scores["Histogram"] = histogram_accuracy
correlation_cal_data["Histogram"] = histogram



#################### ENTROPY FEATURES ####################

###### Sample Entropy ######
sample = x[:, 10]
sample_accuracy = DT(sample.reshape(500,1), y)
accuracy_scores["Sample"] = sample_accuracy
correlation_cal_data["Sample"] = sample



###### Approximate Entropy ######
approximate = x[:, 11]
# approximate_accuracy = DT(approximate.reshape(500,1),y)
# accuracy_scores["Approximate"] = approximate_accuracy
correlation_cal_data["Approximate"] = approximate


###### Spectral Entropy ######
spectral = x[:, 12]
spectral_accuracy = DT(spectral.reshape(500,1),y)
accuracy_scores["Spectral"] = spectral_accuracy
correlation_cal_data["Spectral"] = spectral


###### Permutation Entropy ######
permutation = x[:, 13]
permutation_accuracy = DT(permutation.reshape(500,1),y)
accuracy_scores["Permutation"] = permutation_accuracy
correlation_cal_data["Permutation"] = permutation


###### Singular Value Decomposition Entropy (SVD) ######
svd = x[:, 14]
svd_accuracy = DT(svd.reshape(500,1), y)
accuracy_scores["SVD"] = svd_accuracy
correlation_cal_data["SVD"] = svd


df = pd.DataFrame(correlation_cal_data)
corr = df.corr()
sns.heatmap(corr, annot=True)
# plt.show()



sorted_accuracies = dict(sorted(accuracy_scores.items(), key=lambda x:x[1], reverse=True))



# print()
iteration = 0 
for f, a in sorted_accuracies.items():
    if (iteration < 10):
        if iteration == 0:
            selected_features.append((f,a))
        else:
            score = -1
            feature = None 
            accuracy = None
            for sf, sa in selected_features:
                cor = 1 / (corr[f][sf])
                f_score = (a * abs(cor/10)) / (a + abs(cor/10))
                
                if f_score > score:
                    score = f_score
                    feature = f
                    accuracy = a
            selected_features.append((feature, accuracy))
        iteration+=1


   

top_features = np.concatenate((median.reshape(500,1), ld.reshape(500,1),
                              std.reshape(500,1), ptp.reshape(500,1), 
                              percentile.reshape(500,1), histogram.reshape(500,1),
                              aass.reshape(500,1), permutation.reshape(500,1)), axis=1)


pickle.dump(top_features, open('tex.pkl' , 'wb'))


# x_train, x_test, y_train, y_test = train_test_split(top_features,y,random_state=seed,test_size=0.2)



# km = KMEANS(x_train)

# cluster1 = ClusterIndicesNumpy(0, km.labels_)
# cluster2 = ClusterIndicesNumpy(1, km.labels_)
# cluster3 = ClusterIndicesNumpy(2, km.labels_)


# cluster1_x = x_train[cluster1]
# cluster2_x = x_train[cluster2]
# cluster3_x = x_train[cluster3]



# cluster1_y = y_train[cluster1]
# cluster2_y = y_train[cluster2]
# cluster3_y = y_train[cluster3]



# clf1 = tree.DecisionTreeClassifier()
# clf1.fit(cluster1_x, cluster1_y)


# clf2 = tree.DecisionTreeClassifier()
# clf2.fit(cluster2_x, cluster2_y)


# clf3 = tree.DecisionTreeClassifier()
# clf3.fit(cluster3_x, cluster3_y)



# score = 0
# for (x_data, y_data) in zip(x_test, y_test):
#     closest_cluster = km.predict(x_data.reshape(1,8))
#     if closest_cluster == 0:
#         y_pred = clf1.predict(x_data.reshape(1,8))
#         print("predicted {} - value {}".format(y_pred, y_data))
#         if (y_pred == y_data):
#             score += 1
#     elif closest_cluster == 1:
#         y_pred = clf2.predict(x_data.reshape(1,8))
#         print("predicted {} - value {}".format(y_pred, y_data))
#         if (y_pred == y_data):
#             score += 1
#     else:
#         y_pred = clf3.predict(x_data.reshape(1,8))
#         print("predicted {} - value {}".format(y_pred, y_data))
#         if (y_pred == y_data):
#             score += 1















