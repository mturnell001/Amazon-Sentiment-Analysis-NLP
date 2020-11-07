# imports
import numpy as np
from vectorizer import load_vectors
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib

label_names = ['negative', 'positive']

# load vectors
x_train_vect, x_test_vect, train, test = load_vectors()

# create and train model
lr_classifier = LogisticRegression(C=1.55, penalty='l1', tol=1e-4, max_iter=256, solver='saga', verbose=5, n_jobs=-1)
print('Model created')

lr_classifier.fit(x_train_vect, train.label)
print('Model fit')

# make predictions
lr_predicted = lr_classifier.predict(x_test_vect)
print('Predictions completed')

# display results
print(f'accuracy: {np.mean(lr_predicted == test.label)}')
print(metrics.classification_report(test.label, lr_predicted, target_names=label_names))

# save model
filename = 'models/lr_model.sav'
joblib.dump(lr_classifier, filename)
print(f'Model saved as \'{filename}\'')
