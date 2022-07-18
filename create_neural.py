import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import  to_categorical
import cv2
import os
import numpy

NUM_CLASSES = 6
CLASSES_LIST = ['c', 'fist', 'l', 'okay', 'palm', 'peace']

def load_dataset(name):
  X = []
  Y = []
  dsize = (40, 36)
  for i in range(NUM_CLASSES):
    for (dirname, dirnames, filenames) in os.walk(os.path.join('frames', name, CLASSES_LIST[i])):
      for f in filenames:
        img = cv2.imread(os.path.join(dirname, f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize)
        X.append(img)
        Y.append(i)
  X = numpy.array(X)
  X = X.reshape(X.shape[0], 40, 36, 1)
  Y = numpy.array(Y)
  Y = to_categorical(Y, NUM_CLASSES)
  X = X.astype('float32')
  X /= 255.0
  return X, Y

X_train, Y_train = load_dataset('training')
X_test, Y_test = load_dataset('testing')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(40, 36, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adadelta())

history = model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=2, validation_data=(X_test, Y_test))
model.save('keras_mnist.h5')
