from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data.data_processing import prepare_data
from models.ensemble_models import get_models


OUTPUT_DIR = Path(__file__).resolve().parent / "notebooks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    X, y, feature_columns = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    rows = []
    trained_models = {}
    for model_name, model in get_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rows.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
                "R2": r2_score(y_test, predictions),
            }
        )
        trained_models[model_name] = model

    results = pd.DataFrame(rows).sort_values(by="MAE")
    results.to_csv(
        OUTPUT_DIR / "ensemble_metrics.csv", index=False, encoding="utf-8-sig"
    )

    best_model_name = results.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    importance = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": best_model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)
    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    plt.figure(figsize=(10, 5))
    plt.bar(results["Model"], results["MAE"], color=["#2a9d8f", "#457b9d", "#f4a261", "#e76f51"])
    plt.title("Сравнение ансамблей по MAE")
    plt.ylabel("MAE")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ensemble_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    top_importance = importance.head(10).sort_values(by="Importance")
    plt.barh(top_importance["Feature"], top_importance["Importance"], color="#264653")
    plt.title(f"Топ-10 важных признаков: {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print()
    print("Сравнение ансамблевых моделей:")
    print(results.to_string(index=False))
    print()
    print(f"Лучшая модель по MAE: {best_model_name}")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
