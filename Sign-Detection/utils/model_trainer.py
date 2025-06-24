from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

def build_and_train_model(X_train, y_train, X_test, y_test):
    num_classes = len(set(y_train))

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(1, X_train.shape[2])))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, callbacks=[es])

    # Save model
    model_path = "asl_model.h5"
    model.save(model_path)
    return model_path
