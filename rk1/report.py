from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def prepare_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    df["species_name"] = df["target"].map(
        {
            0: iris.target_names[0],
            1: iris.target_names[1],
            2: iris.target_names[2],
        }
    )

    df["petal_length_category"] = pd.cut(
        df["petal length (cm)"],
        bins=[0, 2.5, 5.0, 10.0],
        labels=["short", "medium", "long"],
        include_lowest=True,
    ).astype("object")

    numeric_missing_idx = [3, 14, 27, 48, 59, 76, 89, 101, 118, 134]
    categorical_missing_idx = [5, 16, 33, 57, 72, 94, 111, 145]

    df.loc[numeric_missing_idx, "sepal width (cm)"] = np.nan
    df.loc[categorical_missing_idx, "petal_length_category"] = np.nan

    return df


def save_histogram(df):
    plt.figure(figsize=(9, 5))
    sns.histplot(df["petal length (cm)"], bins=20, kde=True, color="#457b9d")
    plt.title("Гистограмма признака petal length (cm)")
    plt.xlabel("petal length (cm)")
    plt.ylabel("Количество наблюдений")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "petal_length_hist.png", dpi=150, bbox_inches="tight")
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
    plt.savefig(BASE_DIR / "missing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_data()
    df.to_csv(DATA_DIR / "iris_with_missing.csv", index=False, encoding="utf-8-sig")

    print("Размер исходного набора данных:", df.shape)
    print()
    print("Пропуски до обработки:")
    print(df[["sepal width (cm)", "petal_length_category"]].isna().sum().to_string())

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    processed_df = df.copy()
    processed_df[["sepal width (cm)"]] = numeric_imputer.fit_transform(
        processed_df[["sepal width (cm)"]]
    )
    processed_df[["petal_length_category"]] = categorical_imputer.fit_transform(
        processed_df[["petal_length_category"]]
    )

    processed_df.to_csv(
        DATA_DIR / "iris_processed.csv", index=False, encoding="utf-8-sig"
    )

    before_missing = df[["sepal width (cm)", "petal_length_category"]].isna().sum()
    after_missing = processed_df[
        ["sepal width (cm)", "petal_length_category"]
    ].isna().sum()

    summary = pd.DataFrame(
        {
            "feature": ["sepal width (cm)", "petal_length_category"],
            "type": ["quantitative", "categorical"],
            "missing_before": before_missing.values,
            "missing_after": after_missing.values,
            "imputation_method": ["median", "most_frequent"],
        }
    )
    summary.to_csv(
        BASE_DIR / "missing_processing_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_histogram(df)
    save_missing_plot(before_missing, after_missing)

    description = processed_df.describe(include="all")
    description.to_csv(BASE_DIR / "descriptive_statistics.csv", encoding="utf-8-sig")

    feature_note = {
        "recommended_features": [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
        "optional_feature": "petal_length_category",
        "reason": (
            "Для дальнейшего машинного обучения лучше использовать исходные "
            "количественные признаки, так как они являются первичными измерениями "
            "и содержат максимум информации. Искусственный категориальный признак "
            "petal_length_category построен из petal length (cm), поэтому он "
            "дублирует часть информации и нужен только для экспериментов."
        ),
    }
    with open(BASE_DIR / "feature_selection_note.json", "w", encoding="utf-8") as file:
        json.dump(feature_note, file, ensure_ascii=False, indent=2)

    print()
    print("Использованные методы обработки:")
    print("- Количественный признак sepal width (cm): median")
    print("- Категориальный признак petal_length_category: most_frequent")
    print()
    print("Пропуски после обработки:")
    print(after_missing.to_string())
    print()
    print("Признаки для дальнейшего построения моделей:")
    for feature in feature_note["recommended_features"]:
        print("-", feature)
    print()
    print("Почему:")
    print(feature_note["reason"])


if __name__ == "__main__":
    main()
