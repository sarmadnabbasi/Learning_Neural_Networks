import numpy as np
from keras import layers, optimizers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras.backend as K

K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image

image1 = image.load_img("cat.jpg", target_size=(128, 128))

# plt.imshow(image1)
# plt.show()


# create a data generator
datagen = image.ImageDataGenerator('channels_last')

# load and iterate training dataset
train_it = datagen.flow_from_directory('data/train/', class_mode='binary', batch_size=64, target_size=(100, 100))
# load and iterate test dataset
test_it = datagen.flow_from_directory('data/test/', class_mode='binary', batch_size=64, target_size=(100, 100))
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))


# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# define model
model = define_model()


# Creating model
print("CHeck this : " + str(batchX.shape[1:]))
happyModel = HappyModel(batchX.shape[1:])

# Compile the model
# happyModel.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
opt = optimizers.SGD(lr=0.0009, momentum=0.9)

happyModel.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# train model
happyModel.fit_generator(train_it, epochs=2)

# test model
loss = happyModel.evaluate_generator(test_it)
print()
print("Loss = " + str(loss[0]))
print("Test Accuracy = " + str(loss[1]))
