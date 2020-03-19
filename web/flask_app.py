from flask import Flask, jsonify, render_template
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze(review = ''):
    #key for our data labels
    sentiment_labels = {1:'Negative', 2:'Positive'}

    #sklearn needs an iterable
    review_iter = [review]

    #vectorize for tf-idf methods
    vctrzr = joblib.load('/resources/bin/fitted_vectorizer.sav')
    review_vctr = vctrzr.transform(review_iter)

    #make predictions

    #logistic regression
    lr_model = joblib.load('/resources/bin/top_lr_model.sav')
    lr_label = lr_model.predict(review_vctr)[0]
    lr_prediction = sentiment_labels[lr_label]

    #support vector machine
    svm_model = joblib.load('resources/bin/top_svm_model.sav')
    svm_label = svm_model.predict(review_vctr)[0]
    svm_prediction = sentiment_labels[svm_label]

    #TODO: ADD MODELS HERE!!

    #model vectorization on review
    #model loading joblib.load('top_type_model.sav')
    #model prediction method
    #type_prediction = sentiment_labels[1||2]

    #add the model type and the prediction here

    predictions = {'Review Text':review,
                   'Logistic Regression':lr_prediction,
                   'Support Vector Machine':svm_prediction}
    
    response = jsonify(predictions)
    response.headers.add('Access-Control-Allow-Orogin', '*')
    return response

app = Flask(__name__)

@app.route('/')
def index():
    """
    This renders the home page/base route
    """
    return render(template('index.html'))

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