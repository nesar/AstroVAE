'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference
https://blog.keras.io/building-autoencoders-in-keras.html


 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer, Dropout
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras import optimizers

import SetPub
SetPub.set_pub()

totalFiles = 256
TestFiles = 32

original_dim = 2549 #2551 # mnist ~ 784
intermediate_dim0 = 1024 #
intermediate_dim1 = 512 #
intermediate_dim = 256 #
latent_dim = 10

batch_size = 1
num_epochs = 10 #110 #50
epsilon_std = 1.0 # 1.0
learning_rate = 1e-7
decay_rate = 0.09


# -------------------------------- Network Architecture ------------------------------

x = Input(shape=(original_dim,)) # Deepen encoder after this
h0 = Dense(intermediate_dim0, activation = 'relu')(x) # ADDED intermediate_layer_0
h1 = Dense(intermediate_dim1, activation = 'relu')(h0) # ADDED intermediate_layer_1
h = Dense(intermediate_dim, activation='relu')(h1)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h0)
# h = Dropout(.5)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu') # Deepen decoder after this
decoder_h1 = Dense(intermediate_dim1, activation='relu') # ADDED layer_1
decoder_h0 = Dense(intermediate_dim0, activation='relu') # ADDED layer_0


decoder_mean = Dense(original_dim, activation='sigmoid')

h_decoded = decoder_h(z)
h1_decoded = decoder_h1(h_decoded)
h0_decoded = decoder_h0(h1_decoded)
x_decoded_mean = decoder_mean(h0_decoded)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        # xent_loss = K.sum(metrics.binary_crossentropy(x, x_decoded_mean))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        # return K.mean(xent_loss + 100*kl_loss)
        return K.mean(xent_loss + kl_loss)
        # return K.mean(xent_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
rmsprop = optimizers.RMSprop(lr= learning_rate, rho=0.9, epsilon=None, decay=decay_rate) # Added
# adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate)

vae = Model(x, y)  #Model(input layer, loss function)??
vae.compile(optimizer='rmsprop', loss=None)

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


vae.fit(x_train, shuffle=True, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, None), verbose = 2)
print('---- Training done -------')

#-------------------------------------------------------------------------------

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


# plt.figure(687, figsize=(7, 6))
# plt.title('Encoded outputs')
# plt.xlabel('E[0]')
# plt.ylabel('E[1]')
# CS = plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test[:,0], s = 15, alpha=0.6)
# cbar = plt.colorbar(CS)
# cbar.ax.set_ylabel(r'$\Omega_m$')
# # cbar.ax.set_ylabel(r'$\sigma_8$')
# plt.tight_layout()
# plt.savefig('../Cl_data/Plots/VAE_encodedOutputs_y0.png')



#---------- Saving encoded o/p (latent variables) -------------

# display a 2D plot of the digit classes in the latent space
x_train_encoded = encoder.predict(x_train, batch_size=batch_size)


plt.figure(687, figsize=(7, 6))
plt.title('Encoded outputs')
plt.xlabel('E[0]')
plt.ylabel('E[1]')
CS = plt.scatter(x_train_encoded[:,0], x_train_encoded[:,1], c=y_train[:,0], s = 15, alpha=0.6)
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel(r'$\Omega_m$')
# cbar.ax.set_ylabel(r'$\sigma_8$')
plt.tight_layout()
plt.savefig('../Cl_data/Plots/VAE_encodedOutputs_y0.png')

np.save('../Cl_data/Data/encoded_xtrain_'+str(totalFiles)+'.npy', x_train_encoded)


# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))

_h_decoded = decoder_h(decoder_input)
_h1_decoded = decoder_h1(_h_decoded)    ## ADDED layer_1
_h0_decoded = decoder_h0(_h1_decoded)    ## ADDED --- should replicate decoder arch
_x_decoded_mean = decoder_mean(_h0_decoded)
generator = Model(decoder_input, _x_decoded_mean)
x_decoded = generator.predict(x_train_encoded)



SaveModel = True
if SaveModel:
    epochs = np.arange(1, num_epochs+1)
    train_loss = vae.history.history['loss']
    val_loss = vae.history.history['val_loss']

    training_hist = np.vstack([epochs, train_loss, val_loss])

    # fileOut = 'Stack_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)

    fileOut = 'Model_'+str(totalFiles)
    vae.save('../Cl_data/Model/fullAE_' + fileOut + '.hdf5')
    encoder.save('../Cl_data/Model/Encoder_' + fileOut + '.hdf5')
    generator.save('../Cl_data/Model/Decoder_' + fileOut + '.hdf5')
    np.save('../Cl_data/Model/TrainingHistory_'+fileOut+'.npy', training_hist)



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

plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    # epochs = np.arange(1, epochs+1)
    train_loss = vae.history.history['loss']
    val_loss = vae.history.history['val_loss']


    fig, ax = plt.subplots(1,1, sharex= True, figsize = (8,6))
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epochs,train_loss, '-', lw =1.5)
    ax.plot(epochs,val_loss, '-', lw = 1.5)
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train loss','val loss'])
    plt.tight_layout()
    plt.savefig('../Cl_data/Plots/Training_loss.png')

plt.show()

PlotModel = True
if PlotModel:
    from keras.utils.vis_utils import plot_model
    fileOut = '../Cl_data/Plots/ArchitectureFullAE.png'
    plot_model(vae, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = '../Cl_data/Plots/ArchitectureEncoder.png'
    plot_model(encoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = '../Cl_data/Plots/ArchitectureDecoder.png'
    plot_model(generator, to_file=fileOut, show_shapes=True, show_layer_names=True)

print(x_train_encoded)