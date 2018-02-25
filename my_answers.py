import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = series[window_size:]

    # for loop for dividing series into different time frame windows
    for x in range(series.size - window_size):
        X.append(series[x:x+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size, lstm_units = 5, dropout = 0.2):
    model = Sequential()
    model.add(LSTM(lstm_units, activation='tanh', input_shape=(window_size,1), dropout=dropout, return_sequences=True))
    model.add(LSTM(lstm_units, activation='tanh', dropout=dropout))
    model.add(Dense(1, activation="tanh"))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    text = ''.join([c if should_retain(c) else ' ' for c in text])
    return text


def should_retain(char):
    punctuation = ['!', ',', '.', ':', ';', '?']
    alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    if (char in punctuation) or (char in alpha) or (char == " "):
        return True
    else:
        return False

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    no_complete_window = (len(text) - window_size) // step_size
    last_complete_window = (no_complete_window + 1) * step_size

    for i in range(0, last_complete_window, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars, lstm_units=200, dropout=0.2):
    sher_model = Sequential()
    sher_model.add(LSTM(lstm_units, activation='tanh', input_shape=(window_size,num_chars), dropout=dropout, return_sequences=True))
    sher_model.add(LSTM(lstm_units, activation='tanh', dropout=dropout))
    sher_model.add(Dense(num_chars))
    sher_model.add(Activation('softmax'))

    return sher_model
