import tensorflow as tf 
import os
import keras.backend as K 
import numpy as np 
import matplotlib.pyplot as plt 
from imageio import imread
import matplotlib.image as mpimg

from keras.models import Sequential 
from keras.layers import Conv2D, Dropout
from keras.optimizers import Nadam, Adam  

from keras.preprocessing.image import ImageDataGenerator
import skimage

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# #config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
# #                        allow_soft_placement=True, device_count = {'CPU': 1})
# # config = tf.ConfigProto(intra_op_parallelism_threads=2)
# # session = tf.Session(config=config)
# import keras.backend as K
K.set_floatx('float32')
# K.set_session(session)

# Dog_Dir = '/users/loganjaeger/Desktop/self_python_files/style_transfer/dogs/images'
# Flower_Dir = '/users/loganjaeger/Desktop/self_python_files/style_transfer/flowers'

# flowers = [i for i in os.listdir(Flower_Dir) if os.path.isdir(os.path.join(Flower_Dir, i))]
# dogs = [i for i in os.listdir(Dog_Dir) if os.path.isdir(os.path.join(Dog_Dir, i))]

def gather_ims(Dir, subdir_lst, amount):
	ims = []
	for _ in range(int(amount)):
		subdir = np.random.choice(subdir_lst)
		lst_of_ims = os.listdir(os.path.join(Dir, subdir))
		#print(subdir)
		choice = np.random.choice(lst_of_ims)
		dir_of_choice = os.path.join(Dir, subdir, choice)
		im = r = mpimg.imread(dir_of_choice)
		ims.append(im)
	return ims

def create_data_to_feed(Train_Num, Val_Num, batch_size, dogs, flowers):
	amount = int((Train_Num + Val_Num) / 2)
	t = int(Train_Num / 2)
	v = int(Val_Num / 2)
	dog_ls = [i for i in os.listdir(dogs) if os.path.isdir(os.path.join(dogs, i))]
	flower_ls = [i for i in os.listdir(flowers) if os.path.isdir(os.path.join(flowers, i))]

	dog_array = gather_ims(dogs, dog_ls, amount)
	flower_array = gather_ims(flowers, flower_ls, amount)

	all_array = dog_array + flower_array
	padded = pad_ds(all_array)
	print(padded.shape)
	padded = skimage.measure.block_reduce(padded, (1, 2, 2, 1), func = np.mean)
	print(padded.shape)

	all_dogs = padded[:amount]
	all_flowers = padded[amount:]

	train_dogs = all_dogs[:t]
	val_dogs = all_dogs[t:]

	train_flowers = all_flowers[:t]
	val_flowers = all_flowers[t:]

	train_dog_answers = np.ones(t)
	train_flower_answers = np.zeros(t)

	val_dog_answers = np.ones(v)
	val_flower_answers = np.zeros(v)

	train_dogs = np.array(train_dogs)
	train_flowers = np.array(train_flowers)
	train_data = np.concatenate((train_dogs, train_flowers))
	train_answers = np.concatenate((train_dog_answers, train_flower_answers))

	val_dogs = np.array(val_dogs)
	val_flowers = np.array(val_flowers)
	val_data = np.concatenate((val_dogs, val_flowers))
	val_answers = np.concatenate((val_dog_answers, val_flower_answers))

	train_gen = ImageDataGenerator()
	val_gen = ImageDataGenerator()

	train_generator = train_gen.flow(train_data, train_answers, batch_size=batch_size, seed=89)
	validation_generator = val_gen.flow(val_data, val_answers, batch_size=batch_size, seed=7)

	return (train_generator, validation_generator)

def pad_ds(data):
	s0 = 0
	s1 = 0
	for im in data:
		s = im.shape
		if s[0] > s0:
			s0 = s[0]
		if s[1] > s1:
			s1 = s[1]
	x = []
	for im in data:
		shape = im.shape
		pad_width = ((0, s0 - shape[0]), (0, s1 - shape[1]), (0, 0))
		im = np.pad(im, pad_width, mode = 'constant', constant_values = 0)
		x.append(im)
	return np.array(x)




