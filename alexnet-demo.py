  '''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping

import pickle


batch_size = 128
num_classes = 43
epochs = 30
data_augmentation = False
num_predictions = 20


# TODO: Fill this in based on where you saved the training and testing data

dlxl_file_test = '/home/phung/PycharmProjects/traffic-dl-san/test_cut.pkl'
dlxl_file = '/home/phung/PycharmProjects/traffic-dl-san/train_thu.pkl'
# // load data
with open(dlxl_file_test, mode='rb') as f:
    test_xl = pickle.load(f)
with open(dlxl_file, mode='rb') as f:
    train_xl = pickle.load(f)

x_train , y_train = train_xl['X'], train_xl['Y']
x_test , y_test = test_xl['X'], test_xl['Y']
# y_test = y_test.reshape(12630)
print("Y_shape",y_train.shape)
print("X_text",x_test.shape)
print('Y_test',y_test.shape)


######in data

n_train = x_train.shape[0]

# Number of testing examples.
n_test = x_test.shape[0]

# Find the shape of a traffic sign image
image_shape = x_test.shape[1:]

# Find unique classes/labels in the dataset.


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)


print(x_train.shape)
print(y_train.shape)
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


## xay dung model
locnet = Sequential()
locnet.add(Conv2D(16, (7, 7), padding='valid', input_shape=(32, 32, 3)))
locnet.add(MaxPooling2D(pool_size=(2, 2)))
locnet.add(Conv2D(32, (5, 5), padding='valid'))
locnet.add(MaxPooling2D(pool_size=(2, 2)))
locnet.add(Conv2D(64, (3, 3), padding='valid'))
locnet.add(MaxPooling2D(pool_size=(2, 2)))

locnet.add(Flatten())
locnet.add(Dense(128))
locnet.add(Activation('elu'))
locnet.add(Dense(64))
locnet.add(Activation('elu'))
locnet.add(Dense(num_classes, ))
locnet.add(Activation('softmax'))


# initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0) # co the dung Adadelta
opt =  keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Let's train the model using RMSprop
locnet.compile(loss='categorical_crossentropy', # tinh ham mat mat bang crossentropy
              optimizer=opt,
              metrics=['accuracy'])
# norm du lieu, thu bo
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')

    locnet.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)

locnet.save('traffic-sign-model-cuoi-cung.h5')

scores = locnet.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
