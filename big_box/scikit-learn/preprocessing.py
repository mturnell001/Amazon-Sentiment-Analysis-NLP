#imports
import pandas as pd
from vectorizer import fit_vectorizer, vectorize, fit_scaler, scale

#load data
train = pd.read_csv('../../data/train.csv', names=['label', 'title', 'review'])

#only run this part once!
#fit_vectorizer(train.review)
fit_scaler('bin/x_train_vect.sav')

#may take a while
x_train_filename = 'bin/x_train_scaled.sav'
vectorize(train.review, x_train_filename)


#should go faster
test = pd.read_csv('../../data/test.csv', names=['label', 'title', 'review'])
x_test_filename = 'bin/x_test_scaled.sav'
vectorize(test.review, x_test_filename)