import numpy as np
from vectorizer import load_vectors
from sklearn.svm import LinearSVC
from sklearn import metrics
import joblib

#load vectors
x_train_vect, x_test_vect, train, test = load_vectors()

#create model
lsv_clf = LinearSVC(verbose=5, tol=1e-5, dual=False)

#train model
lsv_clf.fit(x_train_vect, train.label)

#make predictions
lsv_predicted = lsv_clf.predict(x_test_vect)

#display results
print(f'accuracy: {np.mean(lsv_predicted == test.label)}')
print(metrics.classification_report(test.label, lsv_predicted,
                                    target_names=['negative', 'positive']))

joblib.dump(lsv_clf, 'lsv_clf.sav')
print('Model saved as \'lsv_clf.sav\'')