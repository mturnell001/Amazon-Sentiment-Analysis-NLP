#imports
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

def fit_vectorizer(train_data):
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_vectorizer.fit(train_data.review)
    vectorizer_filename = 'bin/fitted_vectorizer.sav'
    joblib.dump(tf_idf_vectorizer, vectorizer_filename)
    return

def fit_scaler(vector_filename):
    vectors = joblib.load(vector_filename)
    max_abs_scaler = MaxAbsScaler()
    max_abs_scaler.fit(vectors)
    joblib.dump(max_abs_scaler, 'bin/fitted_scaler.sav')
    return

def scale(data, filename):
    scaler = joblib.load('bin/fitted_scaler.sav')
    print('Scaling...')
    scaled = scaler.transform(data)
    joblib.dump(scaled, filename)
    print(f'Scaled vectors saved to {filename}')
    return

def vectorize(data, filename):
    tf_idf_vectorizer = joblib.load('bin/fitted_vectorizer.sav')
    print('Vectorizing...')
    vectorized = tf_idf_vectorizer.transform(data)
    scale(vectorized, filename)
    return

def load_vectors(light=False):
    # read in data
    print('Loading Labels...')
    y_test = pd.read_csv('../../data/test.csv', names=['label', 'title', 'review']).drop(columns=['review','title'])
    if(light == False):
        y_train = pd.read_csv('../../data/train.csv', names=['label', 'title', 'review']).drop(columns=['review','title'])

    # load vectors
    print('Loading Vectors...')
    if(light == False):
        x_train = joblib.load('bin/x_train_scaled.sav')
    x_test = joblib.load('bin/x_test_scaled.sav')
    if(light == False):
        return x_train, x_test, y_train, y_test
    else:
        return x_test, y_test

def load_data(light=False):
    print('Loading Data...')
    test = pd.read_csv('../../data/test.csv', names=['label', 'title', 'review']).drop(columns=['title'])
    if(light == False):
        train = pd.read_csv('../../data/train.csv', names=['label', 'title', 'review']).drop(columns=['title'])
        return train, test
    else:
        return test
