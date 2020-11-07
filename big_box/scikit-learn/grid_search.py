from vectorizer import load_vectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

X_test_vect, X_train_vect, test, train = load_vectors()

lr_model = LogisticRegression(n_jobs=-1, C=1.55, max_iter=500, tol=5e-5)
lr_params = dict(penalty=['l1', 'l2', 'none'],
                 solver=['sag', 'saga', 'newton-cg'],
                 )
lr_grid = GridSearchCV(estimator=lr_model, param_grid=lr_params, verbose=5, n_jobs=-1)
lr_search = lr_grid.fit(X_train_vect, train.label)
print(lr_search.best_params_)
print(lr_search.best_score_)