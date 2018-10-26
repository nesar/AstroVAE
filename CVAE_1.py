
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Lambda, merge
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf


m = 32   # minibatch size?
n_z = 2  # dimention of latent space
n_epoch = 30

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels

n_x = X_train.shape[1] # 784
n_y = y_train.shape[1] # 10

print n_x,  n_y

# Q(z|X) -- encoder
X = Input(batch_shape=(m, n_x)) # size of MNIST images
cond = Input(batch_shape=(m, n_y))

inputs = merge([X, cond], mode='concat', concat_axis=1)
h_q = Dense(128, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2.) * eps

z = Lambda(sample_z)([mu, log_sigma])
z_cond = merge([z, cond], mode='concat', concat_axis=1) # <--- NEW!


# P(X|z) -- decoder
decoder_hidden = Dense(128, activation='relu')
decoder_out = Dense(784, activation='sigmoid')

h_p = decoder_hidden(z_cond)
outputs = decoder_out(h_p)



# Overall VAE model, for reconstruction and training
vae = Model([X, cond], outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model([X, cond], mu)

# Generator model, generate new data given latent variable z
d_cond = Input(shape=(n_y,))
d_z = Input(shape=(n_z,))
d_inputs = merge([d_z, d_cond], mode='concat', concat_axis=1)
d_h = decoder_hidden(d_inputs)
d_out = decoder_out(d_h)
decoder = Model([d_z, d_cond], d_out)


def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for eatch data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed from as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl



vae.compile(optimizer='adam', loss=vae_loss)
vae.fit([X_train, y_train], X_train, batch_size=m, nb_epoch=n_epoch, validation_split=0.1)