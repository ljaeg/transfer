import tensorflow as tf 
import keras.backend as K 
import numpy as np 
import matplotlib.pyplot as plt 
from imageio import imread
import matplotlib.image as mpimg
import os

from keras.models import Sequential 
from keras.layers import Conv2D, Dropout, MaxPool2D, GlobalMaxPooling2D, Dense
from keras.optimizers import Nadam, Adam  

#from img_preprocess import create_noise, standardize_img

import vectorize
#hello

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#                        allow_soft_placement=True, device_count = {'CPU': 1})
# config = tf.ConfigProto(intra_op_parallelism_threads=2)
# session = tf.Session(config=config)
import keras.backend as K
K.set_floatx('float32')
# K.set_session(session)

# Dog_Dir = '/users/loganjaeger/Desktop/self_python_files/style_transfer/dogs/images'
# Flower_Dir = '/users/loganjaeger/Desktop/self_python_files/style_transfer/flowers'
Dog_Dir = '/home/admin/Desktop/NST/dogs'
Flower_Dir = '/home/admin/Desktop/NST/flowers'

# flowers = [i for i in os.listdir(Flower_Dir) if os.path.isdir(os.path.join(Flower_Dir, i))]
# dogs = [i for i in os.listdir(Dog_Dir) if os.path.isdir(os.path.join(Dog_Dir, i))]

train_num = 300
val_num = 100
batch_size = 8

train, val = vectorize.create_data_to_feed(train_num, val_num, batch_size, Dog_Dir, Flower_Dir)

x = train[0]
s = x[0].shape

ConvScale = 16
DenseScale = 16
dropout_rate = .4

model = Sequential()
model.add(Conv2D(int(2 * ConvScale), (3, 3), padding = 'valid', activation = 'relu', input_shape = (None, None, 3)))
model.add(Conv2D(int(2 * ConvScale), (3, 3), padding = 'valid', activation = 'relu'))
model.add(MaxPool2D())

model.add(Conv2D(int(ConvScale), (3, 3), padding = 'valid', activation = 'relu'))
model.add(Conv2D(int(ConvScale), (3, 3), padding = 'valid', activation = 'relu'))
model.add(MaxPool2D())

model.add(Conv2D(int(ConvScale), (3, 3), padding = 'valid', activation = 'relu'))
model.add(Conv2D(int(ConvScale), (3, 3), padding = 'valid', activation = 'relu'))
model.add(MaxPool2D())

model.add(Conv2D(int(ConvScale), (3, 3), padding = 'valid', activation = 'relu'))
model.add(GlobalMaxPooling2D())

model.add(Dense(int(2*DenseScale), activation = 'relu'))
model.add(Dropout(dropout_rate))

model.add(Dense(int(2*DenseScale), activation = 'relu'))
model.add(Dropout(dropout_rate))

model.add(Dense(int(DenseScale), activation = 'relu'))
model.add(Dropout(dropout_rate))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = Adam(lr = .0002), loss = 'binary_crossentropy', metrics = ['acc'])
model.summary()

model.fit_generator(
	generator = train,
	epochs = 10,
	verbose = 2,
	validation_data = val,
	steps_per_epoch = np.ceil(train_num / batch_size),
	validation_steps = np.ceil(val_num / batch_size)
	)
