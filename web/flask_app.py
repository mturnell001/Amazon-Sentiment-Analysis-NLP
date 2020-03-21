from flask import Flask, jsonify, render_template
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import os

import spacy
import keras
import tensorflow as tf
import numpy as np
import en_core_web_lg

def analyze(review = ''):
    #key for our data labels
    sentiment_labels = {1:'Negative', 2:'Positive'}

    #sklearn needs an iterable
    review_iter = [review]

    #vectorize for tf-idf methods
    vctrzr = joblib.load(os.path.join('static', 'bin', 'fitted_vectorizer.sav'))
    review_vctr = vctrzr.transform(review_iter)
    preprocessor = vctrzr.build_preprocessor()
    tokenizer = vctrzr.build_tokenizer()
    pre_tokens = preprocessor(review)
    tokens = [' ' + token for token in list(set(tokenizer(pre_tokens)))]

    #make predictions

    #logistic regression
    lr_model = joblib.load(os.path.join('static', 'bin', 'top_lr_model.sav'))
    lr_label = lr_model.predict(review_vctr)[0]
    lr_prediction = sentiment_labels[lr_label]

    #support vector machine
    svm_model = joblib.load(os.path.join('static', 'bin', 'top_svm_model.sav'))
    svm_label = svm_model.predict(review_vctr)[0]
    svm_prediction = sentiment_labels[svm_label]

    #TODO: ADD MODELS HERE!!

    #LSTM requires per-word vectors, the other DNN models act on the centroid vector
    nlp = en_core_web_lg.load()
    PAD_VECTOR = [[0.0] * 300]

    lstm_tokens = []
    for doc in nlp(review):
        lstm_tokens.append(doc.vector)

    #model was trained with length 77, due to the median length of the reviews
    if len(lstm_tokens) < 77:
        lstm_tokens = lstm_tokens + PAD_VECTOR * (77 - len(lstm_tokens))
    elif len(lstm_tokens) > 77:
        lstm_tokens = lstm_tokens[:77]
    else:
        lstm_tokens = lstm_tokens
    lstm_tokens = np.array(lstm_tokens).reshape(-1,77,300)
    #lstm_tokens.shape should be (1,77,300)


    DNN_tokens = nlp(review).vector
    DNN_tokens = DNN_tokens.reshape(1,-1)
    #DNN_tokens.shape will be (1,300) here

    #Untuned LSTM
    lstm_model = tf.keras.models.load_model('static/bin/LSTM_Untuned.h5')
    lstm_label = lstm_model.predict_classes(lstm_tokens)[0]
    lstm_confidence = lstm_model.predict(lstm_tokens)[0][lstm_label]
    lstm_prediction = sentiment_labels[lstm_label + 1]

    #Tuned DNN
    tuned_DNN = tf.keras.models.load_model('static/bin/Hyperas_tuned_DNN.h5')
    tuned_label = tuned_DNN.predict_classes(DNN_tokens)[0]
    tuned_confidence = tuned_DNN.predict(DNN_tokens)[0][tuned_label]
    tuned_prediction = sentiment_labels[tuned_label + 1]

    #Keras Untuned DNN
    untuned_DNN = tf.keras.models.load_model('static/bin/Untuned_DNN.h5')
    untuned_label = untuned_DNN.predict_classes(DNN_tokens)[0]
    untuned_confidence = untuned_DNN.predict(DNN_tokens)[0][untuned_label]
    untuned_prediction = sentiment_labels[untuned_label + 1]

    predictions = {'Tokens':tokens, #jsonify will alpha sort this dict by key
                   'Logistic Regression':lr_prediction,
                   'Support Vector Machine':svm_prediction,
                   'LSTM (Untuned)': f"{lstm_prediction} with {round(lstm_confidence*100,2)}% confidence",
                   'Keras DNN (Tuned)' : f"{tuned_prediction} with {round(tuned_confidence*100,2)}% confidence",
                   'Keras DNN (Untuned)' : f"{untuned_prediction} with {round(untuned_confidence*100,2)}% confidence"}
    
    response = jsonify(predictions)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

app = Flask(__name__)

@app.route('/')
def index():
    """
    This renders the home page/base route
    """
    return render_template('index.html')

@app.route('/api/<review>')
def api(review = ''):
    """
    This function will call the analyzer and return the 
    results
    """
    results = analyze(review)
    return results

if __name__ == '__main__':
    app.run(debug=True)