import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #(MinMaxScaler: weil keine negativen Werte möglich)


def create_windows(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        window = data[i:(i + window_size)]
        X.append(window)
    return np.array(X)


def preprocess_data(df, window_size=24):

    #Imputation
    df['Passanten'] = df['Passanten'].interpolate(method='linear')

    #Skalierung
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Passanten']])

    #Windowing
    X_train = create_windows(df_scaled, window_size)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    return X_train, scaler


