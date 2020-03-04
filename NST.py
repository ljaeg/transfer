import numpy as np 
#from PIL import Image
import matplotlib.pyplot as plt
import h5py
import os
import tensorflow as tf
# config = tf.config.experimental
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# physical_devices = tf.config.list_physical_devices() 
# print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices, True) 
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, GlobalMaxPooling2D, Reshape, UpSampling2D, Flatten, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras import backend
from keras.constraints import MinMaxNorm
from keras.optimizers import RMSprop, Adam
from keras.losses import binary_crossentropy


conv_scale = 8
kernel_size = (3, 3)
dense_scale = 8
Save_dir = "/home/admin/Desktop/transfer/art.h5"
img_save_dir = "/home/admin/Desktop/transfer/new_ims"
real_img_dir = "/home/admin/Desktop/for_transfer_ims/Post-Impressionism.hdf5"

def load_real_samples():
	x = h5py.File(real_img_dir, "r")["images"]
	X = np.array(x).astype("float32")
	#X = np.expand_dims(X, axis = 3)
	y = np.ones(len(x))
	return (X - 127.5) / 127.5, y

def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

def make_discriminator():
	mmn = MinMaxNorm(min_value = -.01, max_value = .01)
	model = Sequential()
	model.add(Conv2D(conv_scale, kernel_size, padding = "same", kernel_constraint = mmn))
	model.add(LeakyReLU(alpha = .2))
	model.add(Conv2D(2*conv_scale, kernel_size, padding = "same", kernel_constraint = mmn))
	model.add(LeakyReLU(alpha = .2))
	model.add(Conv2D(2*conv_scale, kernel_size, padding = "same", kernel_constraint = mmn))
	model.add(BatchNormalization(momentum = .95))
	model.add(LeakyReLU(alpha = .2))
	# model.add(Conv2D(2*conv_scale, kernel_size, padding = "same"))
	# model.add(BatchNormalization(momentum = .8))
	# model.add(LeakyReLU(alpha = .2))
	model.add(Flatten())
	model.add(Dense(1, activation = "linear"))
	model.compile(optimizer = RMSprop(lr = .00005), loss = binary_crossentropy, metrics = ["accuracy"])
	return model

def make_generator(latent_dim = 100):
	model = Sequential()
	model.add(Dense(30 * 25 * 25, activation = "relu", input_shape = (latent_dim,)))
	model.add(Reshape((25, 25, 30)))
	model.add(UpSampling2D(size = (3, 3)))
	model.add(Conv2D(4*conv_scale, kernel_size = kernel_size, padding = "same"))
	model.add(BatchNormalization(momentum = .95))
	model.add(Activation("relu"))
	model.add(UpSampling2D(size = (2, 2)))
	model.add(Conv2D(4*conv_scale, kernel_size = kernel_size, padding = "same"))
	model.add(BatchNormalization(momentum = .95))
	model.add(Activation("relu"))
	model.add(UpSampling2D(size = (2, 2)))
	model.add(Conv2D(4*conv_scale, kernel_size = kernel_size, padding = "same"))
	model.add(BatchNormalization(momentum = .95))
	model.add(Activation("relu"))
	model.add(Conv2D(4*conv_scale, kernel_size = kernel_size, padding = "same"))
	model.add(BatchNormalization(momentum = .95))
	model.add(Activation("relu"))
	model.add(Conv2D(3, kernel_size = (7, 7), padding = "same", activation = "tanh"))
	return model

def make_combined(generator, discriminator):
	discriminator.trainable = False 
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	model.compile(optimizer = RMSprop(lr = .00005), loss = wasserstein_loss)
	return model

def generate_fake_samples(generator, latent_dim, n_samples, noise):
	X = generator.predict(noise)
	y = np.zeros(n_samples) - 1
	return X, y

def save_ims(epoch, generator, latent_dim):
	epoch_number = epoch + 1
	noise = np.random.randn(9 * latent_dim).reshape(9, latent_dim)
	gen_ims = generator.predict(noise)
	for i, im in enumerate(gen_ims, 1):
		im = ((.5 * im) + .5).reshape((300, 300))
		plt.subplot(3, 3, i)
		plt.imshow(im, cmap = "gray")
		plt.axis("off")
	plt.savefig(os.path.join(img_save_dir, "epoch_{}.png".format(epoch_number)))
	plt.close()

def save_best_images(epoch, generator, critic, latent_dim):
	epoch_number = epoch + 1
	noise = np.random.randn(100 * latent_dim).reshape(100, latent_dim)
	gen_ims = generator.predict(noise)
	critic_scores = critic.predict(gen_ims)
	indices = np.arange(len(critic_scores))
	d = dict(zip(indices, critic_scores))
	best_9 = sorted(d, key=d.get, reverse=True)[:9]
	for i, ind in enumerate(best_9, 1):
		im = gen_ims[ind, :, :, 0]
		im = ((.5 * im) + .5)
		plt.subplot(3, 3, i)
		plt.imshow(im, cmap = "gray")
		plt.axis("off")
	plt.savefig(os.path.join(img_save_dir, "epoch_{}.png".format(epoch_number)))
	plt.close()


def train(generator, discriminator, combined, latent_dim = 100, epochs = 150, batch_size = 16, save_interval = 30):
	#load real samples
	real, _ = load_real_samples()

	#perform training for epochs = EPOCHS
	for epoch in range(epochs):
		#Get batch size amount of real images
		idx = np.random.randint(0, real.shape[0], batch_size)
		real_imgs = real[idx]
		real_y = np.ones(batch_size)

		#get batch size amount of generated images
		noise = np.random.randn(latent_dim * batch_size).reshape(batch_size, latent_dim)
		gen_ims, gen_y = generate_fake_samples(generator, latent_dim, batch_size, noise)

		#train discriminator
		d_loss_real, acc_real = discriminator.train_on_batch(real_imgs, real_y)
		d_loss_fake, acc_fake = discriminator.train_on_batch(gen_ims, gen_y)
		d_total_loss = .5 * np.add(d_loss_fake, d_loss_real)

		#train generator
		if not (epoch + 1) % 5:
			g_loss = combined.train_on_batch(noise, gen_y)
			#show progress
			print("epoch {}/{}".format(epoch + 1, epochs))
			print("d_loss_real: {}".format(d_loss_real))
			print("d_loss_fake: {}".format(d_loss_fake))
			print("d_total_loss: {}".format(d_total_loss))
			print("g_loss: {}".format(g_loss))
			print(" ")

		#save ims
		if not (epoch + 1) % save_interval:
			save_best_images(epoch, generator, discriminator, latent_dim)
			#save_ims(epoch, generator, latent_dim)

	#save the generator
	generator.save(Save_dir)


def do():
	gen = make_generator()
	disc = make_discriminator()
	comb = make_combined(gen, disc)
	train(gen, disc, comb, epochs = 5000, batch_size = 8, save_interval = 500)

do()



