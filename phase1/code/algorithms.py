import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_roc
from sklearn.model_selection import KFold, cross_val_score



from sklearn.model_selection import train_test_split

#SVM
from sklearn.svm import SVC

#RF
from sklearn.ensemble import RandomForestClassifier

#KNN
from sklearn.neighbors import KNeighborsClassifier

import random
import os

k_folds = KFold(n_splits = 5)



def SVM(x, y, x_train, x_test, y_train, y_test):
    #https://vitalflux.com/accuracy-precision-recall-f1-score-python-example/

    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)*100
    precision = precision_score(y_test,y_pred)*100
    recall = recall_score(y_test,y_pred)*100
    confusion = confusion_matrix(y_test, y_pred)*100
    svm_scores = cross_val_score(clf, x, np.ravel(y), cv = 5)


    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


    return  {
        "accuracy" : accuracy,
        "precision" : precision,
        "recall" : recall,
        "confusion" : confusion,
        "svm_score" : svm_scores.mean()
    }

def RF(x, y, train_x, train_y, test_x, test_y):

    clf=RandomForestClassifier(max_depth=5, random_state=2)
    clf.fit(train_x, train_y)
    y_pred=clf.predict(test_x)


    accuracy = accuracy_score(test_y,y_pred)
    precision = precision_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    confusion = confusion_matrix(test_y, y_pred)

    rf_scores = cross_val_score(clf, x, np.ravel(y), cv=k_folds)

    fpr, tpr, threshold = roc_curve(test_y, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


    return  {
        "accuracy" : accuracy,
        "precision" : precision,
        "recall" : recall,
        "confusion" : confusion,
        "rf_score" : rf_scores.mean()
    }

def KNN(x, y, train_x, train_y, test_x, test_y):
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='manhattan')
    knn.fit(train_x, train_y)
    y_pred = knn.predict(test_x)

    accuracy = accuracy_score(test_y,y_pred)*100
    precision = precision_score(test_y, y_pred)*100
    recall = recall_score(test_y, y_pred)*100
    confusion = confusion_matrix(test_y, y_pred)

    knn_scores = cross_val_score(knn, x, np.ravel(y), cv=k_folds)


    fpr, tpr, threshold = roc_curve(test_y, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


    return  {
        "accuracy" : accuracy,
        "precision" : precision,
        "recall" : recall,
        "confusion" : confusion,
        "knn_score" : knn_scores.mean()
    }



seed = 57
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


ex = pickle.load(open('nex.pkl', 'rb'))
ey = pickle.load(open('ey.pkl', 'rb'))


# split dataset to train and test
x_train, x_test, y_train, y_test = train_test_split(ex,ey,random_state=seed,test_size=0.2)



#################### SVM ####################
# kernel functions (linear, poly, sigmoid)
svm_result = SVM(ex, ey,x_train, x_test, y_train, y_test)

accuracy = svm_result["accuracy"]
precision = svm_result["precision"]
recall = svm_result["recall"]
confusion = svm_result["confusion"]
cross_validation = svm_result["svm_score"]

result = "\n SVM result \n average cross validation score(mean) : {}\n accuracy score : {}\n precision score : {}\n recall score : {}\n confusion matrix : \n{}\n"
print(result.format(cross_validation, accuracy, precision, recall, confusion))



#################### Random Forest ####################

rf_result = RF(ex, ey, x_train, y_train, x_test, y_test)

accuracy = rf_result["accuracy"]
precision = rf_result["precision"]
recall = rf_result["recall"]
confusion = rf_result["confusion"]
cross_validation = rf_result["rf_score"]

result = "\nRandom Forest result \naverage cross validation score(mean) : {}\naccuracy score : {}\nprecision score : {}\nrecall score : {}\nconfusion matrix : \n{}\n"
print(result.format(cross_validation, accuracy, precision, recall, confusion))


#################### K Neighbors ####################

knn_result = KNN(ex, ey, x_train, y_train, x_test, y_test)

accuracy = knn_result["accuracy"]
precision = knn_result["precision"]
recall = knn_result["recall"]
confusion = knn_result["confusion"]
cross_validation = knn_result["knn_score"]

result = "\nK Neighbors result \naverage cross validation score(mean) : {}\naccuracy score : {}\nprecision score : {}\nrecall score : {}\nconfusion matrix : \n{}\n"
print(result.format(cross_validation, accuracy, precision, recall, confusion))
