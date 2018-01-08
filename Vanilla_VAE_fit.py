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
from keras import optimizers

import SetPub
SetPub.set_pub()

totalFiles = 10000
batch_size = 100
original_dim = 351 # mnist ~ 784
latent_dim = 2
intermediate_dim = 128 # mnist ~ 256
epochs = 10 #110 #50
epsilon_std = 1.0 # 1.0


x = Input(shape=(original_dim,)) # Deepen encoder after this

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
decoder_h = Dense(intermediate_dim, activation='relu') # Deepen decoder after this
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
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)  #Model(input layer, loss function)??

rmsprop = optimizers.RMSprop(lr=1e-6, rho=0.9, epsilon=None, decay=0.01) # Added

vae.compile(optimizer='rmsprop', loss=None)

# ------------------------------------------------------------------------------

import pk_load

density_file = '../Pk_data/SVDvsVAE/Pk5.npy'
halo_para_file = '../Pk_data/SVDvsVAE/Para5.npy'
pk = pk_load.density_profile(data_path = density_file, para_path = halo_para_file)

(x_train, y_train), (x_test, y_test) = pk.load_data()

x_train = x_train[:totalFiles]
x_test = x_test[:np.int(0.2*totalFiles)]
y_train = y_test[:totalFiles]
y_test = y_test[:np.int(0.2*totalFiles)]


print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')

normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
print('-------normalization factor:', normFactor)

x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# ------------------------------------------------------------------------------


vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None), verbose = 2)

#-------------------------------------------------------------------------------

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


plt.figure(687, figsize=(7, 6))
plt.title('Encoded outputs')
plt.xlabel('E[0]')
plt.ylabel('E[1]')
CS = plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test[:,0], s = 15, alpha=0.6)
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel(r'$\Omega_m$')
# cbar.ax.set_ylabel(r'$\sigma_8$')
plt.tight_layout()
# plt.savefig('../Pk_data/SVDvsVAE/VAE_encodedOutputs_y0.png')



#---------- Saving encoded o/p (latent variables) -------------

# display a 2D plot of the digit classes in the latent space
x_train_encoded = encoder.predict(x_train, batch_size=batch_size)

np.save('../Pk_data/SVDvsVAE/encoded_xtrain.npy', x_train_encoded)
np.save('../Pk_data/SVDvsVAE/normfactor.npy', normFactor)


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)
x_decoded = generator.predict(x_train_encoded)



SaveModel = True
if SaveModel:
    epochs = np.arange(1, epochs+1)
    train_loss = vae.history.history['loss']
    val_loss = vae.history.history['val_loss']

    training_hist = np.vstack([epochs, train_loss, val_loss])

    # fileOut = 'Stack_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)

    fileOut = 'Model'
    vae.save('../Pk_data/fullAE_' + fileOut + '.hdf5')
    encoder.save('../Pk_data/Encoder_' + fileOut + '.hdf5')
    generator.save('../Pk_data/Decoder_' + fileOut + '.hdf5')
    np.save('../Pk_data/TrainingHistory_'+fileOut+'.npy', training_hist)



PlotSample = False
k = np.load('../Pk_data/k5.npy')
if PlotSample:
    for i in range(30,31):
        plt.figure(91, figsize=(8,6))
        plt.plot(k, x_decoded[i], 'rx', alpha = 0.2)
        plt.plot(k, x_train[i], 'k')
        plt.xscale('log')
        plt.yscale('log')

    plt.show()


PlotModel = True
if PlotModel:
    from keras.utils.vis_utils import plot_model
    fileOut = '../Pk_data/SVDvsVAE/ArchitectureFullAE.png'
    plot_model(vae, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = '../Pk_data/SVDvsVAE/ArchitectureEncoder.png'
    plot_model(encoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = '../Pk_data/SVDvsVAE/ArchitectureDecoder.png'
    plot_model(generator, to_file=fileOut, show_shapes=True, show_layer_names=True)

print('---- Training done -------')