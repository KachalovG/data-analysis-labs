from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor


def get_model(X_train, y_train):
    model = MultiOutputRegressor(SVR())

    param_grid = {
        'estimator__kernel': ['linear', 'rbf'],
        'estimator__C': [0.1, 1, 10],
        'estimator__epsilon': [0.1, 0.2]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_
