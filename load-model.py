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
from keras.models import load_model
import pickle


batch_size = 128
num_classes = 43
epochs = 1
data_augmentation = False
num_predictions = 20


# TODO: Fill this in based on where you saved the training and testing data

dlxl_file_test = '/home/phung/PycharmProjects/traffic-dl-san/test_cut.pkl'
dlxl_file = '/home/phung/PycharmProjects/traffic-dl-san/train_thu.pkl'
# chay may anh tu
# dlxl_file_test = '/home/quynhpt/traffic-dl-san/test_cut.pkl'
# dlxl_file = '/home/quynhpt/traffic-dl-san/train_thu.pkl'
 
with open(dlxl_file_test, mode='rb') as f:
    test_xl = pickle.load(f)
with open(dlxl_file, mode='rb') as f:
    train_xl = pickle.load(f)

x_train , y_train = train_xl['X'], train_xl['Y']
x_test , y_test = test_xl['X'], test_xl['Y']
# y_test = y_test.reshape(12630)
print("Y_shape",y_train.shape)

# x_train = x_train[:1000, :, :, :]
# y_train = y_train[:1000]
# print("y_shape sau", y_train.shape)
# x_test = x_test[500,:,:,:]
# y_test = y_test[500]
print("X_text",x_test.shape)

print('Y_test',y_test.shape)

# Find the shape of a traffic sign image


# Find unique classes/labels in the dataset.
print("X_train",x_train.shape)
print("y_trai",y_train.shape)
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('Y_train', y_train.shape)
print('X_train',y_test.shape)
############# xay dung model

# ############### load model
model_load = load_model('traffic-sign-model-cuoi-cung.h5')


# print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model_load.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
