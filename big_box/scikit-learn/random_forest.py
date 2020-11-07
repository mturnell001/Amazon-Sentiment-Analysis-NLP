import numpy as np
from vectorizer import load_vectors
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib

#load vectors
x_train_vect, x_test_vect, train, test = load_vectors()

#create model
rf_clf = RandomForestClassifier(bootstrap=True, max_samples=0.4, n_jobs=-1, verbose=9)

#train model
rf_clf.fit(x_train_vect, train.label)

#make predictions
rf_predict = rf_clf.predict(x_test_vect)

#display results
print(f'accuracy: {np.mean(rf_predict == test.label)}')
print(metrics.classification_report(test.label, rf_predict, target_names=['neg', 'pos']))

#save model
joblib.dump(rf_clf, 'models/rf_clf.sav')
print('Model saved as \'rf_clf.sav\'')