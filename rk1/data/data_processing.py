# скачивание датасета
from sklearn.datasets import load_iris
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import numpy as np
from unicodedata import category

iris = load_iris(as_frame=True)
df = iris.frame
df.to_csv('iris.csv', index=False)

#создание категориальных столбцов

df["petal_length_cat"] = pd.cut(
    df["petal length (cm)"],
    bins=[0, 2.5, 5, 10],
    labels=["small", "medium", "large"]
)

df["sepal_width_cat"] = pd.cut(
    df["sepal width (cm)"],
    bins=[0, 2.5, 5, 10],
    labels=["small", "medium", "large"]
)

print(df.head())
df.to_csv('iris2.csv', index=False)

#создание пропусков

np.random.seed(42)
df_with_nan = df.copy()
mask = np.random.rand(*df_with_nan.shape) < 0.2
df_with_nan = df_with_nan.mask(mask)
df_with_nan.to_csv('iris_with_nan.csv', index=False)

print(df_with_nan.isna().sum())#пропуски есть! Датасет испорчен :)

#починка датасета, для котегориальных данных буду использовать вставку по  самому популярному значению, для чисовых MICE

#починка числовых столбцов
num_cols = df_with_nan.select_dtypes(include=np.number).columns
imputer = IterativeImputer(max_iter=10, random_state=42)
imputed_num_array= imputer.fit_transform(df_with_nan[num_cols])
df_imputed_num = pd.DataFrame(imputed_num_array, columns=num_cols)
print(df_imputed_num.isna().sum()) # числовые столбцы заполнены

#починка категориальных столбцов + их дальнейшая кодировка по принципу Ordinal Encoding, тк есть ранжирование
cat_cols = df_with_nan.select_dtypes(include=['object', 'string', 'category', 'bool']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df_with_nan[cat_cols] = cat_imputer.fit_transform(df_with_nan[cat_cols])

category_order = ['small','medium','large']
orders = [category_order] * len(cat_cols)
encoder = OrdinalEncoder(categories=orders)
df_with_nan[cat_cols] = encoder.fit_transform(df_with_nan[cat_cols])
print(df_with_nan[cat_cols].head())

#собрать весь датасет
df_final = pd.concat([df_imputed_num, df_with_nan[cat_cols].reset_index(drop=True)], axis=1)

print("Итоговый датасет без пропусков:")
print(df_final.head())
print("Проверки на пропуски:")
print(df_final.isna().sum())