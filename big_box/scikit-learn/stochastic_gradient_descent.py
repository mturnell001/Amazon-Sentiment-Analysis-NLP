import numpy as np
from vectorizer import load_vectors
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import joblib

#load vectors
x_train_vect, x_test_vect, train, test = load_vectors()

#initialize model
sgd_classifier = SGDClassifier(verbose=5, n_jobs=-1, tol=1e-4)

#train_model
sgd_classifier.fit(x_train_vect, train.label)

#make predictions
sgd_predicted = sgd_classifier.predict(x_test_vect)

#display results
print(f'accuracy: {np.mean(sgd_predicted == test.label)}')
print(metrics.classification_report(test.label, sgd_predicted,
                                    target_names=['negative', 'positive']))

joblib.dump(sgd_classifier, 'models/sgd_classifier.sav')
print('Model saved as \'sgd_classifier.sav\'')
