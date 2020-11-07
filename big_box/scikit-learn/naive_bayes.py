#don't run this one unless you have hundreds of GBs of RAM
import numpy as np
from vectorizer import load_vectors
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
import joblib

#load vectors
x_train_vect, x_test_vect, train, test = load_vectors()

#create model
nb_clf = CategoricalNB()

#train model
nb_clf.fit(x_train_vect.toarray(), train.label)

#make predictions
nb_predict = nb_clf.predict(x_test_vect)

#display results
print(f'accuracy: {np.mean(nb_predict == test.label)}')
print(metrics.classification_report(test.label, nb_predict, target_names=['neg', 'pos']))

#save model
joblib.dump(nb_clf, 'models/nb_clf.sav')
print('Model saved as \'nb_clf.sav\'')