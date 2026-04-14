import pandas as pd
import os

def prepare_data():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'ipc_dataset.csv')
    df = pd.read_csv(file_path, sep=';')
    target_cols = ['total', 'food', 'non_food', 'services']
    month_map = {
        'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4,
        'май': 5, 'июнь': 6, 'июль': 7, 'август': 8,
        'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12
    }
    df['month'] = df['month'].map(month_map)
    for col in target_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)

    for col in target_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        df[f'{col}_lag3'] = df[col].shift(3)
    df = pd.get_dummies(df, columns=['month'], prefix='m')

    df.dropna(inplace=True)

    month_features = [col for col in df.columns if col.startswith('m_')]
    lag_features = [col for col in df.columns if '_lag' in col]
    features = month_features + lag_features

    X = df[features]
    y = df[target_cols]
    return X, y

