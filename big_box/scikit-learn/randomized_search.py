import numpy as np
from vectorizer import load_vectors
from scipy.stats import uniform
from sklearn.utils.fixes import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

X_test_vect, X_train_vect, test, train = load_vectors()

lr_model = LogisticRegression(n_jobs=-1, solver='sag', C=1.55, max_iter=500)
lr_params = dict(tol=loguniform(1e-7, 5e-4))
lr_grid = RandomizedSearchCV(lr_model, lr_params, verbose=5, n_iter=12, n_jobs=-1)
lr_search = lr_grid.fit(X_train_vect, train.label)
print(lr_search.best_params_)
print(lr_search.best_score_)