import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from data_preprocess import getTrainData

train = getTrainData(include_random=True)

X = train['text'].array
y = train['label'].array

idxs = np.array(train.index)

n = 4
kf = KFold(n_splits=n, shuffle=True, random_state=0)

i = 0
accuracy = np.zeros(n)
precision = np.zeros(n)
recall = np.zeros(n)

print('LR:')

for train_index, test_index in kf.split(X):
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]
    
    cv = TfidfVectorizer(strip_accents='ascii', lowercase=True, stop_words='english')
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_val)
    
    lr = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=10, penalty='l2',max_iter=1000)
    lr.fit(X_train_cv, y_train)
    predictions = lr.predict(X_test_cv)
    
    accuracy[i] = accuracy_score(y_val, predictions)
    precision[i] = precision_score(y_val, predictions)
    recall[i] = recall_score(y_val, predictions)
    
    print('Fold index: ', i)
    print('Accuracy score: ', accuracy[i])
    print('Precision score: ', precision[i])
    print('Recall score: ', recall[i])
    i += 1
print('SUM:')
print('Accuracy score: ', np.mean(accuracy))
print('Precision score: ', np.mean(precision))
print('Recall score: ', np.mean(recall))