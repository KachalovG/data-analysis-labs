from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_knn_pipeline(n_neighbors=5, weights="distance", p=2):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)),
        ]
    )
