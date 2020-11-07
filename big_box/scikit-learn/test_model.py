import numpy as np
from sklearn import metrics

# pick model
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#testing models:
from vectorizer import load_vectors
from vectorizer import load_data
import joblib

#load data
test = load_data(light=True)

# load model
print('Loading Model...')
filename ='models/lr_pipeline_8891.sav'
clf = joblib.load(filename)

# make predictions
print(f'Predicting with model: {filename}...')
predicted = clf.predict(test.review)
print(f'{filename}\nAccuracy: {np.mean(predicted == test.label)}')
print(metrics.classification_report(test.label, predicted,
                                    target_names=['negative', 'positive']))

# #for training and testing model types:
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
#
# X, y = make_classification(n_samples=3600, n_features=94)
# X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y)
# clf = MLPClassifier(early_stopping=True,verbose=4, n_iter_no_change=4)
# clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
# print(clf.score(X_test, y_test))
# print(f'Model Accuracy: {np.mean(predicted == y_test)}')
# print(metrics.classification_report(y_test, predicted))