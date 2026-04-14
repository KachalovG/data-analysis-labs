from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.data_processing import prepare_data
from models.decision_tree_model import get_model as get_tree_model
from models.linear_regression_model import get_model as get_linear_model
from models.svm_kernel_model import get_model as get_svm_kernel_model


OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Decision Tree": get_tree_model(X_train_scaled, y_train),
        "Linear Regression": get_linear_model(X_train_scaled, y_train),
        "SVM": get_svm_kernel_model(X_train_scaled, y_train),
    }

    rows = []
    for model_name, model in models.items():
        predictions = model.predict(X_test_scaled)
        rows.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, predictions),
                "R2": r2_score(y_test, predictions),
            }
        )

    results = pd.DataFrame(rows).sort_values(by="R2", ascending=False)
    results.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    print(results.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(results["Model"], results["MAE"], color=["#2a9d8f", "#e9c46a", "#e76f51"])
    axes[0].set_title("Сравнение моделей по MAE")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(results["Model"], results["R2"], color=["#264653", "#8ab17d", "#f4a261"])
    axes[1].set_title("Сравнение моделей по R2")
    axes[1].set_ylabel("R2")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
