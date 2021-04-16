import time
import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D


def cnn():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # one-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # scale image RGB values to [0, 1]
    X_train = X_train / 255
    X_test = X_test / 255

    # expand dimensions for 2d convolutional filters
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    # build model
    model = cnn()

    # start training
    start_time = time.time()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128, verbose=1)
    print('training took {} seconds'.format(time.time() - start_time))

    # final evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('val accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
