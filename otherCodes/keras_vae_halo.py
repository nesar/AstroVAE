'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference
https://blog.keras.io/building-autoencoders-in-keras.html


 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 29 # mnist ~ 784
latent_dim = 2
intermediate_dim = 10 # mnist ~ 256
epochs = 110 #110 #50
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't act
        # ually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)

# ------------------------------------------------------------------------------

# train the VAE on MNIST digits
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

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


# ------------------------------------------------------------------------------



vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))


# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test[:,1])
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 10  # figure with 10x10 digits
digit_size = 29
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        # digit = x_decoded[0].reshape(digit_size, digit_size)
        # figure[i * digit_size: (i + 1) * digit_size,
        #        j * digit_size: (j + 1) * digit_size] = x_decoded

        plt.figure(30, figsize=(8,6))
        plt.plot(x_decoded[0], 'b', alpha = 0.5)
        # plt.plot()


for i in range(x_test.shape[0]):
    plt.figure(30, figsize=(8,6) )
    plt.plot(x_test[i,:], 'r', alpha = 0.1)

plt.show()

plt.figure(100)
loss = vae.history.history['loss']
val_loss = vae.history.history['val_loss']
plt.plot( np.arange(len(loss)), np.array(loss),  'o-')
plt.plot( np.arange(len(val_loss)), np.array(val_loss),  'o-')
plt.show()
