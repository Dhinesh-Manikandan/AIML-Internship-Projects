import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # Assuming last column is label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    # Reshape X for LSTM: (samples, time_steps=1, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder
