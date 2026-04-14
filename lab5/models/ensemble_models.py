from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor


def get_models(random_state=42):
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, random_state=random_state, n_jobs=1
        ),
        "AdaBoost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4, random_state=random_state),
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }
