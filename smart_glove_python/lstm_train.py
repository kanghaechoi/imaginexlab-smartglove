import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


def lstm_train(X_train, y_train, X_test, y_test):

    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    # truncate and pad input sequences
    max_review_length = 500
    # create the model
    embedding_vecor_length = 32

    model = Sequential()
    model.add(LSTM(100))
    model.add(Dense(50, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(np.shape(X_train))
    model.fit(X_train, y_train, epochs=3)
    print(model.summary())
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))