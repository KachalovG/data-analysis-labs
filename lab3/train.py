from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    TimeSeriesSplit,
    train_test_split,
)

from data.data_processing import prepare_data
from models.knn_model import make_knn_pipeline


RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "notebooks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_params(params):
    normalized = {}
    for key, value in params.items():
        if hasattr(value, "item"):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def evaluate_model(model, X_train, X_test, y_train, y_test, label):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "Model": label,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions),
    }


def main():
    X, y, _ = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    baseline_model = make_knn_pipeline(n_neighbors=5, weights="distance", p=2)
    baseline_metrics = evaluate_model(
        baseline_model, X_train, X_test, y_train, y_test, "Baseline KNN (K=5)"
    )

    param_grid = {
        "knn__n_neighbors": list(range(2, 26)),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    random_distributions = {
        "knn__n_neighbors": np.arange(2, 26),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    cv_strategies = {
        "KFold": KFold(n_splits=5, shuffle=False),
        "TimeSeriesSplit": TimeSeriesSplit(n_splits=5),
    }

    search_rows = []
    best_candidates = []

    for cv_name, cv_strategy in cv_strategies.items():
        grid_search = GridSearchCV(
            estimator=make_knn_pipeline(),
            param_grid=param_grid,
            cv=cv_strategy,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        grid_search.fit(X_train, y_train)
        grid_metrics = evaluate_model(
            grid_search.best_estimator_,
            X_train,
            X_test,
            y_train,
            y_test,
            f"GridSearchCV + {cv_name}",
        )
        grid_row = {
            "Search": "GridSearchCV",
            "CV": cv_name,
            "BestParams": json.dumps(
                normalize_params(grid_search.best_params_), ensure_ascii=False
            ),
            "CV_MAE": -grid_search.best_score_,
            **grid_metrics,
        }
        search_rows.append(grid_row)
        best_candidates.append((grid_search.best_estimator_, grid_row))

        randomized_search = RandomizedSearchCV(
            estimator=make_knn_pipeline(),
            param_distributions=random_distributions,
            n_iter=20,
            cv=cv_strategy,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
        randomized_search.fit(X_train, y_train)
        random_metrics = evaluate_model(
            randomized_search.best_estimator_,
            X_train,
            X_test,
            y_train,
            y_test,
            f"RandomizedSearchCV + {cv_name}",
        )
        random_row = {
            "Search": "RandomizedSearchCV",
            "CV": cv_name,
            "BestParams": json.dumps(
                normalize_params(randomized_search.best_params_), ensure_ascii=False
            ),
            "CV_MAE": -randomized_search.best_score_,
            **random_metrics,
        }
        search_rows.append(random_row)
        best_candidates.append((randomized_search.best_estimator_, random_row))

    search_results = pd.DataFrame(search_rows).sort_values(by="CV_MAE")
    search_results.to_csv(
        OUTPUT_DIR / "search_results_summary.csv", index=False, encoding="utf-8-sig"
    )

    best_estimator, best_row = min(best_candidates, key=lambda item: item[1]["CV_MAE"])
    final_metrics = evaluate_model(
        best_estimator, X_train, X_test, y_train, y_test, "Best tuned KNN"
    )

    comparison = pd.DataFrame([baseline_metrics, final_metrics])
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )

    mae_by_k = []
    for k in range(2, 26):
        model = make_knn_pipeline(n_neighbors=k, weights="distance", p=2)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, f"K={k}")
        mae_by_k.append({"K": k, "MAE": metrics["MAE"], "R2": metrics["R2"]})
    mae_by_k_df = pd.DataFrame(mae_by_k)
    mae_by_k_df.to_csv(OUTPUT_DIR / "mae_by_k.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.plot(mae_by_k_df["K"], mae_by_k_df["MAE"], marker="o", color="#1d3557")
    plt.title("KNN: зависимость MAE от числа соседей")
    plt.xlabel("K")
    plt.ylabel("MAE")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mae_by_k.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(
        comparison["Model"],
        comparison["MAE"],
        color=["#457b9d", "#e76f51"],
    )
    plt.title("Сравнение базовой и оптимальной модели KNN по MAE")
    plt.ylabel("MAE")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "knn_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(OUTPUT_DIR / "best_search.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_search": {
                    "search": best_row["Search"],
                    "cv": best_row["CV"],
                    "best_params": json.loads(best_row["BestParams"]),
                    "cv_mae": best_row["CV_MAE"],
                },
                "baseline_metrics": baseline_metrics,
                "best_metrics": final_metrics,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print()
    print("Базовая модель:")
    print(pd.DataFrame([baseline_metrics]).to_string(index=False))
    print()
    print("Результаты подбора гиперпараметров:")
    print(search_results.to_string(index=False))
    print()
    print("Лучшая стратегия:")
    print(
        json.dumps(
            {
                "Search": best_row["Search"],
                "CV": best_row["CV"],
                "BestParams": json.loads(best_row["BestParams"]),
                "CV_MAE": best_row["CV_MAE"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print("Сравнение baseline и лучшей модели:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
