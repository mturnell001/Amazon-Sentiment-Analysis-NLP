#imports
from vectorizer import load_vectors
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import joblib

#load vectors
x_train, x_test, train, test = load_vectors()

#create model
print('Creating model...')
mlp_classifier = MLPClassifier(verbose=True, early_stopping=True,
                               n_iter_no_change=4)

#train model
print('Training model...')
mlp_classifier.fit(x_train, train.label)

#save model
joblib.dump(mlp_classifier, 'models/nn_mlp.sav')
print('Model saved as \'nn_mlp.sav\'')

#make predictions
print('Making predictions...')
mlp_predicted = mlp_classifier.predict(x_test)


#display results
print(f'accuracy: {np.mean(mlp_predicted == test.label)}')
print(metrics.classification_report(test.label, mlp_predicted,
                                    target_names=['negative', 'positive']))

