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

batch_size = 256
original_dim = 351 # mnist ~ 784
latent_dim = 7
intermediate_dim = 128 # mnist ~ 256
epochs = 4 #110 #50
epsilon_std = 1.0 # 1.0


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

import pk_load

density_file = '../Pk_data/Pk.npy'
halo_para_file = '../Pk_data/Para9.npy'
pk = pk_load.density_profile(data_path = density_file, para_path = halo_para_file)

(x_train, y_train), (x_test, y_test) = pk.load_data()



print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')



x_train = x_train.astype('float32')/28000 #/ 255.
x_test = x_test.astype('float32')/28000 #/ 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# ------------------------------------------------------------------------------
#
plt.figure(34)
plt.plot(np.load('../Pk_data/Pk.npy')[100])
plt.plot(x_train[100], 'x')
plt.yscale('log')
plt.xscale('log')

vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))


# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(65, figsize=(6, 6))
plt.scatter(x_test_encoded[:, 1], x_test_encoded[:, 0], c=y_test[:,0])
plt.colorbar()
plt.show()

plt.figure(66, figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 2], c=y_test[:,1])
plt.colorbar()
plt.show()

plt.figure(67, figsize=(6, 6))
plt.scatter(x_test_encoded[:, 1], x_test_encoded[:, 3], c=y_test[:,2])
plt.colorbar()
plt.show()


plt.figure(687, figsize=(6, 6))
plt.scatter(y_test[:, 3], y_test[:, 4], c=x_test_encoded[:,1])
plt.colorbar()
plt.show()


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 10  # figure with 10x10 digits
digit_size = original_dim
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian

grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):

z_sample = np.array([np.random.uniform(size = latent_dim)])
# z_sample = np.expand_dims(z_sample, axis=1)
x_decoded = generator.predict(z_sample)
plt.figure(91, figsize=(8,6))
plt.plot(x_decoded[0], 'bx', alpha = 0.5)



PlotSample = True
if PlotSample:
    for i in range(10):
        plt.figure(91, figsize=(8,6))
        plt.plot(x_test[i,:], 'r', alpha = 0.7)

    plt.show()

PlotLoss = True
if PlotLoss:
    plt.figure(100)
    loss = vae.history.history['loss']
    val_loss = vae.history.history['val_loss']
    plt.plot( np.arange(len(loss)), np.array(loss),  'o-')
    plt.plot( np.arange(len(val_loss)), np.array(val_loss),  'o-')
    plt.show()
