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


# m = 1 # 50
# n_z = 2

original_dim = 2549 #2551 # mnist ~ 784
intermediate_dim = 512 #

latent_dim = 6
n_epoch = 30 #10
totalFiles = 128
TestFiles = 32


batch_size = 1
epochs = 1 #110 #50
epsilon_mean = 0.0 # 1.0
epsilon_std = 1.0 # 1.0
# learning_rate = 1e-7
# decay_rate = 0.09



# Q(z|X) -- encoder
inputs = Input(shape=(original_dim,))
h_q = Dense(intermediate_dim, activation='relu')(inputs)
mu = Dense(latent_dim, activation='linear')(h_q)
log_sigma = Dense(latent_dim, activation='linear')(h_q)

# -------------------------------------------------------------

def sample_z(args):
    mu, log_sigma = args
    ###eps = K.random_normal(shape=(m, n_z), mean=0., std=1.)
    eps = K.random_normal(shape=(batch_size, latent_dim), mean=epsilon_mean, stddev=epsilon_std)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])

# -------------------------------------------------------------

# P(X|z) -- decoder
decoder_hidden = Dense(latent_dim, activation='relu')
decoder_out = Dense(original_dim, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

# -------------------------------------------------------------


# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu)

# Generator model, generate new data given latent variable z
d_in = Input(shape=(latent_dim,))
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


# ----------------------------- i/o ------------------------------------------

import Cl_load

# density_file = '../Cl_data/Cl_'+str(nsize)+'.npy'
# density_file = '../Cl_data/LatinCl_'+str(nsize)+'.npy'
train_path = '../Cl_data/Data/LatinCl_'+str(totalFiles)+'.npy'
train_target_path =  '../Cl_data/Data/LatinPara5_'+str(totalFiles)+'.npy'
test_path = '../Cl_data/Data/LatinCl_'+str(TestFiles)+'.npy'
test_target_path =  '../Cl_data/Data/LatinPara5_'+str(TestFiles)+'.npy'

# halo_para_file = '../Cl_data/Para5_'+str(nsize)+'.npy'
# halo_para_file = '../Cl_data/LatinPara5_'+str(nsize)+'.npy'

# pk = pk_load.density_profile(data_path = density_file, para_path = halo_para_file)

camb_in = Cl_load.cmb_profile(train_path = train_path,  train_target_path = train_target_path , test_path = test_path, test_target_path = test_target_path, num_para=5)


(x_train, y_train), (x_test, y_test) = camb_in.load_data()

x_train = x_train[:,2:]
x_test = x_test[:,2:]

print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')


# meanFactor = np.mean( [np.mean(x_train), np.mean(x_test ) ])
# print('-------mean factor:', meanFactor)
# x_train = x_train.astype('float32') - meanFactor #/ 255.
# x_test = x_test.astype('float32') - meanFactor #/ 255.
# np.save('../Cl_data/Data/meanfactor_'+str(totalFiles)+'.npy', meanFactor)
#


normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
# normFactor = np.mean( [np.std(x_train), np.std(x_test ) ])
print('-------normalization factor:', normFactor)
x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.
np.save('../Cl_data/Data/normfactor_'+str(totalFiles)+'.npy', normFactor)


x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))



# ------------------------------------------------------------------------------

#TRAIN       -- NaN losses Uhhh

vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(x_train, x_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=2, validation_data=(x_test,
                                                                                      x_test))

# -------------------------------------------------------------

# y_pred = encoder.predict(x_train[10:20,:])

# display a 2D plot of the digit classes in the latent space
plt.figure(figsize=(6, 6))
x_train_encoded = encoder.predict(x_train)
plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c=y_train[:, 0], cmap='spring')
plt.colorbar()

x_test_encoded = encoder.predict(x_test)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test[:, 0], cmap='copper')
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


x_train_encoded = encoder.predict(x_train)
x_decoded = decoder.predict(x_train_encoded)


PlotSample = True
ls = np.load('../Cl_data/Data/ls_'+str(totalFiles)+'.npy')[2:]
if PlotSample:
    for i in range(3,10):
        plt.figure(91, figsize=(8,6))
        plt.plot(ls, x_decoded[i], 'r--', alpha = 0.8)
        plt.plot(ls, x_train[i], 'b--')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.title('reconstructed - red')

    plt.show()