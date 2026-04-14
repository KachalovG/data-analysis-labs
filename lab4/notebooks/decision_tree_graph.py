from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.data_processing import prepare_data
from models.decision_tree_model import get_model as get_tree_model


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

    tree_model = get_tree_model(X_train_scaled, y_train)
    predictions = tree_model.predict(X_test_scaled)

    print(
        pd.DataFrame(
            [
                {
                    "Model": "Decision Tree",
                    "MAE": mean_absolute_error(y_test, predictions),
                    "R2": r2_score(y_test, predictions),
                }
            ]
        ).to_string(index=False)
    )

    feature_importance = (
        pd.Series(tree_model.feature_importances_, index=X.columns)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance.plot(kind="barh", ax=ax, color="#2a9d8f")
    ax.set_title("Важность признаков дерева решений")
    ax.set_xlabel("Важность")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        tree_model,
        feature_names=X.columns,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3,
        ax=ax,
    )
    ax.set_title("Дерево решений (первые 4 уровня)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "decision_tree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
