from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


def get_model(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)

    param_grid = {
        'max_depth': [3, None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'criterion': ['squared_error', 'absolute_error'],
        'max_features': ['sqrt', 'log2', None],
        'max_leaf_nodes': [None, 5, 10, 15, 20],

    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=1,
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_
