#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation


if __name__ == "__main__":
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))

    model.compile(loss="categorial_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
    