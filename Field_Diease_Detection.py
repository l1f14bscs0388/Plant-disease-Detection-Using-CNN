import pickle
import cv2
from os import listdir
from keras.layers.normalization import batch_normalization
from keras.layers import BatchNormalization
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
try:
    from tensorflow.keras.optimizers import Adam
    #from keras.optimizers import Adam
except:
    from keras.optimizers import adam_v2
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.keras.applications.vgg16 import preprocess_input
import os
IMAGE_SIZE = [256, 256]
EPOCHS = 100
INIT_LR = 1e-3
BATCH = 32
chanDim = -1
default_image_size = tuple((256, 256))
image_size = 0
directory_root = './PlantVillage'
Training_Path = directory_root +'/Train'
Validation_Path = directory_root +"/Validate"
total_train = 0
total_valid = 0
for r in os.listdir(Training_Path):
    total_train += len(os.listdir(Training_Path+'/'+r))
for r in os.listdir(Validation_Path):
    total_valid += len(os.listdir(Validation_Path+'/'+r))
width=256
height=256
depth=3
inputShape = (height, width, depth)
n_classes = len(os.listdir(Training_Path))
RANDOM_SEED = 1
def RGB_to_HSV(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.25],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function= RGB_to_HSV
    #preprocessing_function= preprocess_input
)
test_datagen = ImageDataGenerator(
    preprocessing_function= RGB_to_HSV,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.25],
    horizontal_flip=True,
    vertical_flip=True#, 
    #preprocessing_function= preprocess_input
)
train_generator = train_datagen.flow_from_directory(
    Training_Path,color_mode='rgb',target_size=IMAGE_SIZE,batch_size=BATCH,    class_mode='categorical',    seed=RANDOM_SEED)
validation_generator = test_datagen.flow_from_directory(
    Validation_Path,
    color_mode='rgb',
    target_size=IMAGE_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    seed=RANDOM_SEED
)
model = Sequential()

model.add(Conv2D(256, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.15))

model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
#ssh -4 username@neptun.cs.uni-kl.de, be<Pi7ci, nice -n 10 your_program
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(da(0.25))
model.add(Flatten())

model.add(Dense(512))
model.add(Activation("relu"))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
history = model.fit (
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=total_train // BATCH,
    epochs=EPOCHS, verbose=1
    )
model.save("modelnew.h5")
model.save_weights('model_weights.h5')