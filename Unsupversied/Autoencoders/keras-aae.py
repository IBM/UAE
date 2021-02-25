from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization 
from keras import initializers
from keras.utils import plot_model
from keras.datasets import mnist
from keras.optimizers import Adam
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from datetime import datetime
from setup_mnist import MNIST
from setup_svhn import SVHN
from sklearn.manifold import TSNE
from absl import flags
from absl import app
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


FLAGS = flags.FLAGS

# General
flags.DEFINE_bool("adversarial", True, "Use Adversarial Autoencoder or regular Autoencoder")
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("reconstruct", False, "Reconstruct image")
flags.DEFINE_bool("generate", False, "Generate image from latent")
flags.DEFINE_bool("generate_grid", False, "Generate grid of images from latent space (only for 2D latent)")
flags.DEFINE_bool("plot", False, "Plot latent space")
flags.DEFINE_integer("latent_dim", 2, "Latent dimension")

# Train
flags.DEFINE_integer("epochs", 50, "Number of training epochs")
flags.DEFINE_integer("train_samples", 110000, "Number of training samples from MNIST")
flags.DEFINE_integer("batchsize", 100, "Training batchsize")
flags.DEFINE_string("datasets", "MNIST", "Datasets")

# Test
flags.DEFINE_integer("test_samples", 10000, "Number of test samples from MNIST")
flags.DEFINE_list("latent_vec", None, "Latent vector (use with --generate flag)")



def create_model(input_dim, latent_dim, verbose=False, save_graph=False):
	autoencoder_input = Input(shape=(input_dim,))
	generator_input = Input(shape=(input_dim,))
	
	encoder = Sequential()
	encoder.add(Dense(1000, input_shape=(input_dim,), activation='relu'))
	encoder.add(Dense(1000, activation='relu'))
	encoder.add(Dense(latent_dim, activation='sigmoid'))
	
	decoder = Sequential()
	decoder.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
	decoder.add(Dense(1000, activation='relu'))
	decoder.add(Dense(input_dim,activation='sigmoid'))

	if FLAGS.adversarial:
		discriminator = Sequential()
		discriminator.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
		discriminator.add(Dense(1000, activation='relu'))
		discriminator.add(Dense(1, activation='sigmoid'))

	autoencoder = Model(autoencoder_input, decoder(encoder(autoencoder_input)))
	autoencoder.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error")
	
	if FLAGS.adversarial:
		discriminator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
		discriminator.trainable = False
		generator = Model(generator_input, discriminator(encoder(generator_input)))
		generator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy")
	
	if verbose:
		print("Autoencoder Architecture")
		print(autoencoder.summary())
		if FLAGS.adversarial:
			print("Discriminator Architecture")
			print(discriminator.summary())
			print("Generator Architecture")
			print(generator.summary())

	if save_graph:
		plot_model(autoencoder, to_file="autoencoder_graph.png")
		if FLAGS.adversarial:
			plot_model(discriminator, to_file="discriminator_graph.png")
			plot_model(generator, to_file="generator_graph.png")

	if FLAGS.adversarial:
		return autoencoder, discriminator, generator, encoder, decoder
	else:
		return autoencoder, None, None, encoder, decoder

def train(n_samples, batch_size, n_epochs):
	if FLAGS.datasets == "MNIST":
		autoencoder, discriminator, generator, encoder, decoder = create_model(input_dim=784, latent_dim=FLAGS.latent_dim)
		data = MNIST()
	else:
		autoencoder, discriminator, generator, encoder, decoder = create_model(input_dim=3072, latent_dim=FLAGS.latent_dim)
		data = SVHN()
	
	x_train = data.train_data
	y_train = np.tile(np.argmax(data.train_labels,1),[2])
	print("training data:", x_train.shape,y_train.shape)
	x_test = data.test_data
	y_test = data.test_labels
	
	if FLAGS.datasets == "MNIST":
		x_train = x_train.reshape(-1, 784)
		x_train1 = x_train1.reshape(-1,784)
	else:
		
		x_train = x_train.reshape(-1,3072)
		x_train1 = x_train1.reshape(-1,3072)
	#normalize = colors.Normalize(0., 255.)
	#x = normalize(x)

	
	

	past = datetime.now()
	s=10000
	for epoch in np.arange(1, n_epochs + 1):
		autoencoder_losses = []
		if FLAGS.adversarial:
			discriminator_losses = []
			generator_losses = []
		
		rand_x = np.random.RandomState(42)
		rand_x.shuffle(x_train)
		rand_x = np.random.RandomState(42)
		rand_x.shuffle(x_train1)
		rand_y = np.random.RandomState(42)
		rand_y.shuffle(y_train)
		
		for batch in np.arange(len(x_train) / batch_size):
			start = int(batch * batch_size)
			end = int(start + batch_size)
			if FLAGS.datasets == "SVHN" and batch == np.arange(len(x_train) / batch_size)[-1]:
				samples = x_train[-100:]
				#samples1 = x_train1[-100:]
			else:
				samples = x_train[start:end]
				#samples1 = x_train1[start:end]
			autoencoder_history = autoencoder.fit(x=samples, y=samples, epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
			if FLAGS.adversarial:
				fake_latent = encoder.predict(samples)
				discriminator_input = np.concatenate((fake_latent, np.random.randn(batch_size, FLAGS.latent_dim) ))
				discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
				discriminator_history = discriminator.fit(x=discriminator_input, y=discriminator_labels, epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
				generator_history = generator.fit(x=samples, y=np.ones((batch_size, 1)), epochs=1, batch_size=batch_size, validation_split=0.0, verbose=0)
			
			autoencoder_losses.append(autoencoder_history.history["loss"])
			if FLAGS.adversarial:
				discriminator_losses.append(discriminator_history.history["loss"])
				generator_losses.append(generator_history.history["loss"])
		now = datetime.now()
		print("\nEpoch {}/{} - {:.1f}s".format(epoch, n_epochs, (now - past).total_seconds()))
		print("Autoencoder Loss: {}".format(np.mean(autoencoder_losses)))
		if FLAGS.adversarial:
			print("Discriminator Loss: {}".format(np.mean(discriminator_losses)))
			print("Generator Loss: {}".format(np.mean(generator_losses)))
		past = now
		'''
		if np.mean(autoencoder_losses) < s:
			s = np.mean(autoencoder_losses)
			encoder.save('models/MNIST/AAE/gaussian/0.01/{}_encoder_'+str(num).format(desc))
			decoder.save('models/MNIST/AAE/gaussian/0.01/{}_decoder_'+str(num).format(desc))
			print("--------------best scores------------:", s)
		'''
		'''
		if epoch % 50 == 0:
			print("\nSaving models...")
			# autoencoder.save('{}_autoencoder.h5'.format(desc))
			encoder.save('models/SVHN/{}_encoder_optil2.h5'.format(desc))
			decoder.save('models/SVHN/{}_decoder_optil2.h5'.format(desc))
			# if FLAGS.adversarial:
			# 	discriminator.save('{}_discriminator.h5'.format(desc))
			#generator.save('{}_generator.h5'.format(desc))
		'''
	# autoencoder.save('{}_autoencoder.h5'.format(desc))
	encoder.save('models/MNIST/AE/AAE/gaussian/0.001/{}_encoder'.format(desc))
	decoder.save('models/MNIST/AE/AAE/gaussian/0.001/{}_decoder'.format(desc))
	#generator.save('{}_generator_aug1.h5'.format(desc))
	# if FLAGS.adversarial:
		# discriminator.save('{}_discriminator.h5'.format(desc))
		# generator.save('{}_generator.h5'.format(desc))

def reconstruct(n_samples):
	encoder = load_model('{}_encoder.h5'.format(desc))
	decoder = load_model('{}_decoder.h5'.format(desc))
	data = MNIST()
	x_test = data.test_data
	choice = np.random.choice(np.arange(n_samples))
	original = x_test[choice].reshape(1, 784)
	#normalize = colors.Normalize(0., 255.)
	#original = normalize(original)
	latent = encoder.predict(original)
	reconstruction = decoder.predict(latent)
	draw([{"title": "Original", "image": original}, {"title": "Reconstruction", "image": reconstruction}])

def generate(latent=None):
	decoder = load_model('{}_decoder.h5'.format(desc))
	if latent is None:
		latent = np.random.randn(1, FLAGS.latent_dim)
	else:
		latent = np.array(latent)
	sample = decoder.predict(latent.reshape(1, FLAGS.latent_dim))
	draw([{"title": "Sample", "image": sample}])

def draw(samples):
	fig = plt.figure(figsize=(5 * len(samples), 5))
	gs = gridspec.GridSpec(1, len(samples))
	for i, sample in enumerate(samples):
		ax = plt.Subplot(fig, gs[i])
		ax.imshow((sample["image"] * 255.).reshape(28, 28), cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		ax.set_title(sample["title"])
		fig.add_subplot(ax)
	plt.show(block=False)
	raw_input("Press Enter to Exit")

def generate_grid(latent=None):
	decoder = load_model('{}_decoder.h5'.format(desc))
	samples = []
	for i in np.arange(400):
		latent = np.array([(i % 20) * 1.5 - 15., 15. - (i / 20) * 1.5])
		samples.append({
			"image": decoder.predict(latent.reshape(1, FLAGS.latent_dim))
		})
	draw_grid(samples)

def draw_grid(samples):
	fig = plt.figure(figsize=(15, 15))
	gs = gridspec.GridSpec(20, 20, wspace=-.5, hspace=0)
	for i, sample in enumerate(samples):
		ax = plt.Subplot(fig, gs[i])
		ax.imshow((sample["image"] * 255.).reshape(28, 28), cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('equal')
		# ax.set_title(sample["title"])
		fig.add_subplot(ax)
	plt.show(block=False)
	raw_input("Press Enter to Exit")
	# fig.savefig("images/{}_grid.png".format(desc), bbox_inches="tight", dpi=300)

def plot(n_samples):
	encoder = load_model('{}_encoder.h5'.format(desc))
	data = MNIST()
	x_test = data.test_data
	y_test = data.test_labels
	x = x_test[:n_samples].reshape(n_samples, 784)
	y = y_test[:n_samples]
	#normalize = colors.Normalize(0., 255.)
	#x = normalize(x)
	latent = encoder.predict(x)
	if FLAGS.latent_dim > 2:
		tsne = TSNE()
		print("\nFitting t-SNE, this will take awhile...")
		latent = tsne.fit_transform(latent)
	fig, ax = plt.subplots()
	for label in np.arange(10):
		ax.scatter(latent[(y_test == label), 0], latent[(y_test == label), 1], label=label, s=3)
	ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	ax.set_aspect('equal')
	ax.set_title("Latent Space")
	plt.show(block=False)
	raw_input("Press Enter to Exit")
	# fig.savefig("images/{}_latent.png".format(desc), bbox_inches="tight", dpi=300)

def main(argv):
	global desc
	if FLAGS.adversarial:
		desc = "aae"
	else:
		desc = "regular"
	if FLAGS.train:
		train(n_samples=FLAGS.train_samples, batch_size=FLAGS.batchsize, n_epochs=FLAGS.epochs)
	elif FLAGS.reconstruct:
		reconstruct(n_samples=FLAGS.test_samples)
	elif FLAGS.generate:
		if FLAGS.latent_vec:
			assert len(FLAGS.latent_vec) == FLAGS.latent_dim, "Latent vector provided is of dim {}; required dim is {}".format(len(FLAGS.latent_vec), FLAGS.latent_dim)
			generate(FLAGS.latent_vec)
		else:
			generate()
	elif FLAGS.generate_grid:
		generate_grid()
	elif FLAGS.plot:
		plot(FLAGS.test_samples)


if __name__ == "__main__":
	app.run(main)
