from vectorizer import load_data
import numpy as np
from sklearn import metrics
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import joblib

# load data
train, test = load_data()

# create pipeline
clf = LogisticRegression(C=1.55, penalty ='elasticnet', max_iter=256,
			             solver='saga', n_jobs=-1, verbose=4,
                         multi_class='ovr', l1_ratio=0.5, tol=5e-4
                         )
pipeline = pipeline.make_pipeline(TfidfVectorizer(), MaxAbsScaler(), clf)

print('Fitting Pipeline...')
# fit pipeline
pipeline.fit(train.review, train.label)

# save model
filename = 'lr_pipeline.sav'
joblib.dump(pipeline, filename)
print(f'Model saved as \'{filename}\'')

# make predictions
print('Predicting...')
predicted = pipeline.predict(test.review)
print(f'{filename}\nAccuracy: {np.mean(predicted == test.label)}')
print(metrics.classification_report(test.label, predicted,
                                    target_names=['negative', 'positive']))