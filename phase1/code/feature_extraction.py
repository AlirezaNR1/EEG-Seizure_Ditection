import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.svm import SVC

import random
import os

import entropy as ent
import math
from sklearn.model_selection import KFold

from sklearn import tree




seed = 57
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


x = pickle.load(open('x.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

x_normal = np.concatenate((x[:300], x[400:]), axis=0) #before 300-400
x_seizure = x[300:400]

sampling_freq = 173.6 #based on info from website

b, a = butter(3, [0.5,40], btype='bandpass',fs=sampling_freq)


x_normal_filtered = np.array([lfilter(b,a,x_normal[ind,:]) for ind in range(x_normal.shape[0])])
x_seizure_filtered = np.array([lfilter(b,a,x_seizure[ind,:]) for ind in range(x_seizure.shape[0])])


x_normal = x_normal_filtered
x_seizure = x_seizure_filtered

x = np.concatenate((x_normal,x_seizure))
y = np.concatenate((np.zeros((400,1)),np.ones((100,1))))





#################### TIME DOMAIN FEATURES ####################

###### PTP ###### *
time_domain_PTP = []
for col in x:
    max = np.amax(col)
    min = np.amin(col)
    ptp = max - min
    time_domain_PTP.append(ptp)

time_domain_PTP = np.array(time_domain_PTP)
time_domain_PTP = time_domain_PTP.reshape(500, 1)
# time_domain_PTP = normalization(time_domain_PTP)




###### AASS ######
time_domain_AASS = []
size = x[0].size
for col in x:
    dif = 0 
    for row in range(0,size-1):
        dif += abs(col[row + 1] - col[row])
    dif = dif / size 
    time_domain_AASS.append(dif)

time_domain_AASS = np.array(time_domain_AASS)
time_domain_AASS = time_domain_AASS.reshape(500,1)
# time_domain_AASS = normalization(time_domain_AASS)





###### SSA ###### *
#Singular spectrum analysis
time_domain_SSA = []
size = x[0].size
limit = 5
for col in x:
    sum = 0
    for index in range(limit, size-limit):
        x1 = col[index]
        x2 = col[index - limit]
        x3 = col[index + limit]
        x2p = abs(x2)
        x3p = abs(x3)
        if x2p - x1 == 0 and x3p - x1 == 0:
            sum += (1/2)*abs(((x2-x1)/0.000001) + ((x3-x1)/0.000001)) 
        elif x2p - x1 == 0:
            sum += (1/2)*abs(((x2-x1)/0.000001) + ((x3-x1)/(x3p-x1)))
        elif x3p - x1 == 0:
            sum += (1/2)*abs(((x2-x1)/(x2p-x1)) + ((x3-x1)/0.000001)) 
        else:
            sum += (1/2)*abs(((x2-x1)/(x2p-x1)) + ((x3-x1)/(x3p-x1))) 
    time_domain_SSA.append(sum)

time_domain_SSA = np.array(time_domain_SSA)
time_domain_SSA = time_domain_SSA.reshape(500,1)
# time_domain_SSA = normalization(time_domain_SSA)



###### Log Detect ######
# bug : https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log
np.seterr(divide = 'ignore') 
time_domain_LD = []
size = x[0].size
for col in x:
    sum = 0
    for row in col:
        pos_val = abs(row)
        log = np.log10(pos_val)
        sum += log 
    sum /= size 
    res = math.exp(sum) 
    time_domain_LD.append(res)

time_domain_LD = np.array(time_domain_LD)
time_domain_LD = time_domain_LD.reshape(500,1)
# time_domain_LD = normalization(time_domain_LD)
np.seterr(divide = 'warn') 



###### Zero Crossings(ZC) ###### *
time_domain_ZC = []
size = x[0].size
epsilon = 0.01
for col in x:
    count = 0
    for index in range(0, size-1):
        x1 = col[index]
        x2 = col[index + 1]
        if (x1 > 0 and x2 < 0) or (x1 < 0 and x2 > 0):
            if (abs(x1 - x2) >= epsilon):
                count+=1
    time_domain_ZC.append(count)

time_domain_ZC = np.array(time_domain_ZC)
time_domain_ZC = time_domain_ZC.reshape(500,1)
# time_domain_ZC = normalization(time_domain_ZC)





#################### STATISTICAL FEATURES ####################

###### Mean ######
statistical_mean = []
for col in x:
    mean = np.mean(col)
    statistical_mean.append(mean)

statistical_mean = np.array(statistical_mean)
statistical_mean = statistical_mean.reshape(500,1)
# statistical_mean = normalization(statistical_mean)


# ###### Median ######
statistical_median = []
for col in x:
    median = np.median(col)
    statistical_median.append(median)

statistical_median = np.array(statistical_median)
statistical_median = statistical_median.reshape(500,1)
# statistical_median = normalization(statistical_median)



###### Percentile ###### *
statistical_percentile = []
for col in x:
    percentile = np.percentile(col, 80)
    statistical_percentile.append(percentile)

statistical_percentile = np.array(statistical_percentile)
statistical_percentile = statistical_percentile.reshape(500,1)
# statistical_percentile = normalization(statistical_percentile)


# ###### Standard Derivation(STD) ###### *
statistical_std = []
for col in x:
    std = np.std(col)
    statistical_std.append(std)

statistical_std = np.array(statistical_std)
statistical_std = statistical_std.reshape(500,1)
# statistical_std = normalization(statistical_std)




###### Histogram ######
#https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
statistical_histogram = []
for col in x:
    hist, bin_edges = np.histogram(col, bins='auto', density=True)
    sum = hist.sum()
    statistical_histogram.append(sum)

statistical_histogram = np.array(statistical_histogram)
statistical_histogram = statistical_histogram.reshape(500,1)
# statistical_histogram = normalization(statistical_histogram)




#################### ENTROPY FEATURES ####################
# https://raphaelvallat.com/entropy/build/html/index.html


###### Sample Entropy ######
# https://raphaelvallat.com/entropy/build/html/generated/entropy.sample_entropy.html#entropy.sample_entropy
entropy_sample = []
for col in x:
    res = ent.sample_entropy(col)
    entropy_sample.append(res)

entropy_sample = np.array(entropy_sample)
entropy_sample = entropy_sample.reshape(500,1)
# entropy_sample = normalization(entropy_sample)


###### Approximate Entropy ######
# https://raphaelvallat.com/entropy/build/html/generated/entropy.app_entropy.html#entropy.app_entropy
entropy_approximate = []
for col in x:
    res = ent.app_entropy(col)
    entropy_approximate.append(res)

entropy_approximate = np.array(entropy_approximate)
entropy_approximate = entropy_approximate.reshape(500,1)
# entropy_approximate = normalization(entropy_approximate)



###### Spectral Entropy ######
#https://raphaelvallat.com/entropy/build/html/generated/entropy.spectral_entropy.html
entropy_spectral = []
for col in x:
    sample_freq = np.mean(col)
    res = ent.spectral_entropy(col, sf=sample_freq, method='welch')
    entropy_spectral.append(res)

entropy_spectral = np.array(entropy_spectral)
entropy_spectral = entropy_spectral.reshape(500,1)
# entropy_spectral = normalization(entropy_spectral)


    
###### Permutation Entropy ######
# https://raphaelvallat.com/entropy/build/html/generated/entropy.perm_entropy.html#entropy.perm_entropy
entropy_permutation = []
for col in x:
    res = ent.perm_entropy(col)
    entropy_permutation.append(res)

entropy_permutation = np.array(entropy_permutation)
entropy_permutation = entropy_permutation.reshape(500,1)
# entropy_permutation = normalization(entropy_permutation)



###### Singular Value Decomposition Entropy (SVD) ######
# https://raphaelvallat.com/entropy/build/html/generated/entropy.svd_entropy.html#entropy.svd_entropy
entropy_svd = []
for col in x:
    res = ent.svd_entropy(col)
    entropy_svd.append(res)

entropy_svd = np.array(entropy_svd)
entropy_svd = entropy_svd.reshape(500,1)
# entropy_svd = normalization(entropy_svd)


extracted_features = np.concatenate((time_domain_PTP, time_domain_AASS, 
                                     time_domain_SSA, time_domain_LD, 
                                     time_domain_ZC, statistical_mean,
                                     statistical_median, statistical_percentile,
                                     statistical_std, statistical_histogram,
                                     entropy_sample, entropy_approximate,
                                     entropy_spectral, entropy_permutation,
                                     entropy_svd),axis=1)
print(extracted_features.shape)
print(y.shape)

pickle.dump(extracted_features, open('ex.pkl' , 'wb'))
pickle.dump(y, open('ey.pkl', 'wb'))









