import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.backend import clear_session
import streamlit as st

def load_npz(file):
    data = np.load(file)
    if 'images' in data and 'labels' in data:
        return data['images'], data['labels']
    else:
        return None, None

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

@st.cache_resource(show_spinner="Training...")
def fit_model(X, y, test_size=0.2, epochs=10, num_classes=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    y_train_ohe = to_categorical(y_train, num_classes=num_classes)
    y_test_ohe = to_categorical(y_test, num_classes=num_classes)

    clear_session()
    set_random_seed(42)

    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    model = create_model(X_train.shape[1:], num_classes)

    train_history = model.fit(X_train, y_train_ohe, epochs=epochs)
    test_history = model.fit(X_test, y_test_ohe, epochs=epochs)

    train_accuracy = train_history.history['accuracy'][-1] * 100
    test_accuracy = test_history.history['accuracy'][-1] * 100

    return model, train_history, test_history, train_accuracy, test_accuracy