"""

Followed from https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

"""

from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf


m = 10 # 50
n_z = 2
n_epoch = 30 #10


# Q(z|X) -- encoder
inputs = Input(shape=(29,))
h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

# -------------------------------------------------------------

def sample_z(args):
    mu, log_sigma = args
    ###eps = K.random_normal(shape=(m, n_z), mean=0., std=1.)
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])

# -------------------------------------------------------------

# P(X|z) -- decoder
decoder_hidden = Dense(10, activation='relu')
decoder_out = Dense(29, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

# -------------------------------------------------------------


# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu)

# Generator model, generate new data given latent variable z
d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

# -------------------------------------------------------------
#CUSTOM LOSS 

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl

#-------------------------------------------------------------
# LOAD 
# from keras.datasets import mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# X_train = x_train.astype('float32') / 255.
# ## X_test = x_test.astype('float32') / 255.
# X_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# ## X_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
# X_train = mnist.train.images
# X_train = X_train.astype('float32') / 255.
#
# X_test = mnist.test.images
# X_test = X_test.astype('float32') / 255.
# Y_test = mnist.test.labels
# -------------------------------------------------------------


# train the VAE on halo profile

import halo_load

density_file = '../Halo_data/fof-064-d_profile.npy'
halo_para_file1 = '../Halo_data/fof-064-m_200.npy'
halo_para_file2 = '../Halo_data/fof-064-r_200.npy'
dens = halo_load.density_profile(data_path = density_file, para_path1 = halo_para_file1, para_path2 = halo_para_file2)

(x_train, y_train), (x_test, y_test) = dens.load_data()



print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')



x_train = x_train.astype('float32') #/ 255.
x_test = x_test.astype('float32') #/ 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))




# -------------------------------------------------------------
#TRAIN       -- NaN losses Uhhh

vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(x_train, x_train, batch_size=m, nb_epoch=n_epoch)

# -------------------------------------------------------------

y_pred = encoder.predict(x_train[10:20,:])

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=m)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test[:, 1])
plt.colorbar()
plt.show()

# score = vae.evaluate(X_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


plt.figure(100)
loss = vae.history.history['loss']
# val_loss = vae.history.history['val_loss']
plt.plot( np.arange(len(loss)), np.array(loss),  'o-')
# plt.plot( np.arange(len(val_loss)), np.array(val_loss),  'o-')
plt.show()
