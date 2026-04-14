from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer


OUTPUT_DIR = Path(__file__).resolve().parent / "notebooks"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(
        {
            0: iris.target_names[0],
            1: iris.target_names[1],
            2: iris.target_names[2],
        }
    )

    df["petal_length_group"] = pd.cut(
        df["petal length (cm)"],
        bins=[0, 2.5, 5.0, 10.0],
        labels=["short", "medium", "long"],
        include_lowest=True,
    ).astype("object")

    numeric_missing_idx = [3, 14, 27, 48, 59, 76, 89, 101, 118, 134]
    category_missing_idx = [5, 16, 33, 57, 72, 94, 111, 145]
    df.loc[numeric_missing_idx, "sepal width (cm)"] = np.nan
    df.loc[category_missing_idx, "petal_length_group"] = np.nan

    return df


def save_histogram(df):
    plt.figure(figsize=(9, 5))
    sns.histplot(df["petal length (cm)"], bins=20, kde=True, color="#457b9d")
    plt.title("Гистограмма признака petal length (cm)")
    plt.xlabel("petal length (cm)")
    plt.ylabel("Количество наблюдений")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "petal_length_hist.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_missing_plot(before_missing, after_missing):
    summary = pd.DataFrame(
        {
            "До обработки": before_missing,
            "После обработки": after_missing,
        }
    )
    summary.plot(kind="bar", figsize=(8, 5), color=["#e76f51", "#2a9d8f"])
    plt.title("Количество пропусков до и после обработки")
    plt.ylabel("Число пропусков")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    df = build_dataset()
    df.to_csv(DATA_DIR / "iris_with_missing.csv", index=False, encoding="utf-8-sig")

    before_missing = df[["sepal width (cm)", "petal_length_group"]].isna().sum()

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    processed_df = df.copy()
    processed_df[["sepal width (cm)"]] = numeric_imputer.fit_transform(
        processed_df[["sepal width (cm)"]]
    )
    processed_df[["petal_length_group"]] = categorical_imputer.fit_transform(
        processed_df[["petal_length_group"]]
    )

    after_missing = processed_df[["sepal width (cm)", "petal_length_group"]].isna().sum()

    processed_df.to_csv(
        DATA_DIR / "iris_processed.csv", index=False, encoding="utf-8-sig"
    )

    summary = pd.DataFrame(
        {
            "feature": [
                "sepal width (cm)",
                "petal_length_group",
            ],
            "type": [
                "quantitative",
                "categorical",
            ],
            "missing_before": before_missing.values,
            "missing_after": after_missing.values,
            "imputation_method": [
                "median",
                "most_frequent",
            ],
        }
    )
    summary.to_csv(
        OUTPUT_DIR / "missing_processing_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_histogram(df)
    save_missing_plot(before_missing, after_missing)

    numeric_describe = processed_df.describe(include="all")
    numeric_describe.to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv", encoding="utf-8-sig"
    )

    selected_features = {
        "recommended_features": [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
        "optional_feature_for_experiments": ["petal_length_group"],
        "reason": (
            "Для дальнейшего машинного обучения целесообразно использовать "
            "исходные количественные признаки, потому что они являются первичными "
            "измерениями и содержат максимальный объём информации. Искусственный "
            "категориальный признак petal_length_group построен на основе petal length "
            "и дублирует часть информации, поэтому его лучше не использовать вместе "
            "с исходным petal length в основной модели."
        ),
    }

    with open(OUTPUT_DIR / "feature_selection_note.json", "w", encoding="utf-8") as file:
        json.dump(selected_features, file, ensure_ascii=False, indent=2)

    print("Размер исходного набора данных:", df.shape)
    print()
    print("Пропуски до обработки:")
    print(before_missing.to_string())
    print()
    print("Использованные методы обработки:")
    print("- Количественный признак sepal width (cm): median")
    print("- Категориальный признак petal_length_group: most_frequent")
    print()
    print("Пропуски после обработки:")
    print(after_missing.to_string())
    print()
    print("Рекомендуемые признаки для дальнейшего ML:")
    for feature in selected_features["recommended_features"]:
        print("-", feature)
    print()
    print("Комментарий:")
    print(selected_features["reason"])


if __name__ == "__main__":
    main()
