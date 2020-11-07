import joblib
import en_core_web_lg
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# load Data
print('Loading Data...')
train = pd.read_csv('../../data/train.csv', names=['label', 'title', 'review']).drop(columns=['title'])
test = pd.read_csv('../../data/test.csv', names=['label', 'title', 'review']).drop(columns=['title'])

# shift 1,2 = neg,pos to 0,1 = neg,pos for to_categorical
print('Processing y...')
y_train = train.label.replace(1,0).replace(2,1)
y_test = test.label.replace(1,0).replace(2,1)

# categorical-ize labels
print('to_categorical-izing y...')
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

#load vectorizer
print('Loading vectorizer...')
spacy_vctrzr = en_core_web_lg.load()

#vectorize train and test data
print('Vectorizing training data...')
x_train = [doc.vector for doc in spacy_vctrzr.pipe(train.review, disable=['tagger', 'parser', 'ner'])]

print('Vectorizing testing data...')
x_test = [doc.vector for doc in spacy_vctrzr.pipe(test.review, disable=['tagger', 'parser', 'ner'])]

# scale vectors
print('Scaling x...')
scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

#save data ready for models
print('Saving data ready for models...')
joblib.dump(scaler, 'bin/scaler.sav')
joblib.dump(x_train_scaled,'bin/x_train_scaled.sav')
joblib.dump(x_test_scaled, 'bin/x_test_scaled.sav')
joblib.dump(y_train_cat, 'bin/y_train_cat.sav')
joblib.dump(y_test_cat, 'bin/y_test_cat.sav')