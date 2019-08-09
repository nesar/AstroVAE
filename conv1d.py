from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Keras uses TensforFlow backend as default
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv1D,UpSampling1D
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda, Conv1D
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
# import argparse
import os
import h5py
import GPy

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def rescale(params):
    """
    Rescales parameters between -1 and 1.
    Input :
    - params : physical parameters
    Output :
    - params_new : rescaled parameters
    - theta_mean, theta_mult : rescaling factors
    """
    theta_mean = np.mean(params, axis=0)
    theta_mult = np.max(params - theta_mean, axis=0)
    return (params - theta_mean) * theta_mult**-1, theta_mean, theta_mult


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
     Arguments
        args (tensor): mean and log of variance of Q(z|X)
     Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim), mean=epsilon_mean, stddev=epsilon_std)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
        Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

#
# ################ edits ########################
# # MNIST dataset
# # (x_train1, y_train1), (x_test1, y_test1) = mnist.load_data()
# # image_size1 = x_train1.shape[1]
# # x_train1 = np.reshape(x_train1, [-1, image_size1, image_size1, 1])
# # x_test1 = np.reshape(x_test1, [-1, image_size1, image_size1, 1])
#
# # Load training/testing set
# DataDir = '../../Data/'
# x_train = np.array(h5py.File(DataDir + 'output_cosmos/cosmos_train_512.hdf5', 'r')['galaxies'])
# x_test = np.array(h5py.File(DataDir + 'output_cosmos/cosmos_test_64.hdf5', 'r')['galaxies'])
#
# # y_train = np.loadtxt(DataDir + 'lhc_512_5.txt')
# # y_test = np.loadtxt(DataDir + 'lhc_64_5_testing.txt')
#
# # x_train = Trainfiles[:, num_para+2:]
# # x_test = Testfiles[:, num_para+2:]
# # y_train = Trainfiles[:, 0: num_para]
# # y_test =  Testfiles[:, 0: num_para]
#
# print(x_train.shape, 'train sequences')
# print(x_test.shape, 'test sequences')
# # print(y_train.shape, 'train sequences')
# # print(y_test.shape, 'test sequences')
#
#
# # Rescaling
# xmin = np.min(x_train)
# xmax = np.max(x_train) - xmin
# x_train = (x_train - xmin) / xmax
# x_test = (x_test - xmin) / xmax
#
# # y_train, ymean, ymult = rescale(y_train)
# # y_test = (y_test - ymean) * ymult**-1
#
# # print(y_train)
# # print('----')
# # print(y_test)
#
# # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*, x_train.shape[2]))
# # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
#
# print(x_train.shape)
# print(x_test.shape)
#
# # print(x_train1.shape)
# # print(x_test1.shape)
#
# x_train = K.cast_to_floatx(x_train)
# x_test = K.cast_to_floatx(x_test)
#
# ################################################
# image_size = x_train.shape[1]
# x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
# x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
#
# ################################################
#
#
# x_train = x_train.astype('float32')  # / 255
# x_test = x_test.astype('float32')  # / 255
#
#
#
# ################
# ################



# ----------------------------- i/o ------------------------------------------


import params_debug as params
# import Cl_load
# import SetPub
# SetPub.set_pub()


############### Setting same float, random seeds ##############

np.random.seed(42)
from tensorflow import set_random_seed
set_random_seed(42)
K.set_floatx('float32')

###################### PARAMETERS ##############################

original_dim = params.original_dim # 2549
intermediate_dim3 = params.intermediate_dim3 # 1600
intermediate_dim2 = params.intermediate_dim2 # 1024
intermediate_dim1 = params.intermediate_dim1 # 512
intermediate_dim0 = params.intermediate_dim0 # 256
intermediate_dim = params.intermediate_dim # 256
latent_dim = params.latent_dim # 10

ClID = params.ClID
num_train = params.num_train # 512
num_test = params.num_test # 32
num_para = params.num_para # 5


batch_size = params.batch_size # 8
num_epochs = params.num_epochs # 100
epsilon_mean = params.epsilon_mean # 1.0
epsilon_std = params.epsilon_std # 1.0
learning_rate = params.learning_rate # 1e-3
decay_rate = params.decay_rate # 0.0

noise_factor = params.noise_factor # 0.00

######################## I/O ##################################

DataDir = params.DataDir
PlotsDir = params.PlotsDir
ModelDir = params.ModelDir

fileOut = params.fileOut



Trainfiles = np.loadtxt(DataDir + 'P'+str(num_para)+ClID+'Cl_'+str(num_train)+'.txt')
Testfiles = np.loadtxt(DataDir + 'P'+str(num_para)+ClID+'Cl_'+str(num_test)+'.txt')

x_train = Trainfiles[:, num_para+2:]
x_test = Testfiles[:, num_para+2:]
y_train = Trainfiles[:, 0: num_para]
y_test =  Testfiles[:, 0: num_para]

print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')

ls = np.loadtxt(DataDir+'P'+str(num_para)+'ls_'+str(num_train)+'.txt')[2:]

#------------------------- SCALAR parameter for rescaling -----------------------
#### ---- All the Cl's are rescaled uniformly #####################

'''
minVal = np.min( [np.min(x_train), np.min(x_test ) ])
meanFactor = 1.1*minVal if minVal < 0 else 0
# meanFactor = 0.0
print('-------mean factor:', meanFactor)
x_train = x_train - meanFactor #/ 255.
x_test = x_test - meanFactor #/ 255.

# x_train = np.log10(x_train) #x_train[:,2:] #
# x_test =  np.log10(x_test) #x_test[:,2:] #

normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
# normFactor = 1
print('-------normalization factor:', normFactor)
x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.


np.savetxt(DataDir+'meanfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt', [meanFactor])
np.savetxt(DataDir+'normfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt', [normFactor])

'''
#########  New l-dependant rescaling (may not work with LSTMs) ###########
#### - --   works Ok with iid assumption ############


minVal = np.min( [np.min(x_train, axis = 0), np.min(x_test , axis = 0) ], axis=0)
#meanFactor = 1.1*minVal if minVal < 0 else 0
# meanFactor = 0.0
meanFactor = 1.1*minVal
print('-------mean factor:', meanFactor)
x_train = x_train - meanFactor #/ 255.
x_test = x_test - meanFactor #/ 255.

# x_train = np.log10(x_train) #x_train[:,2:] #
# x_test =  np.log10(x_test) #x_test[:,2:] #

normFactor = np.max( [np.max(x_train, axis = 0), np.max(x_test, axis = 0 ) ], axis = 0)
# normFactor = 1
print('-------normalization factor:', normFactor)
x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.


np.savetxt(DataDir+'meanfactorPArr'+str(num_para)+ClID+'_'+ fileOut +'.txt', [meanFactor])
np.savetxt(DataDir+'normfactorPArr'+str(num_para)+ClID+'_'+ fileOut +'.txt', [normFactor])


##############################################
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# # Trying to get x_train ~ (-1, 1) -- doesn't work well
# x_mean = np.mean(x_train, axis = 0)
# x_train = x_train - x_mean
# x_test = x_test - x_mean


## ADD noise
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# plt.plot(x_test_noisy.T, 'r', alpha = 0.3)
# plt.plot(x_test_noisy.T*(y_test[:,2]**2), 'b', alpha = 0.3)

x_train_noisy = K.cast_to_floatx(x_train_noisy)
x_train = K.cast_to_floatx(x_train)
# ------------------------------------------------------------------------------

x_train = np.expand_dims(x_train, 1)
x_test = np.expand_dims(x_test, 1)






# Input image dimensions
steps, original_dim = 1, original_dim # Take care here since we are changing this according to the data
# Number of convolutional filters to use
filters = 64
# Convolution kernel size
num_conv = 6
# Set batch size
# batch_size = 100
# Decoder output dimensionality
# decOutput = 10

# latent_dim = 20
# intermediate_dim = 256
# epsilon_std = 1.0
epochs = 5

# x = Input(batch_shape=(batch_size,steps,original_dim))
x = Input(shape=(steps,original_dim))

# Play around with padding here, not sure what to go with.
conv_1 = Conv1D(1,
                kernel_size=num_conv,
                padding='same',
                activation='relu')(x)
conv_2 = Conv1D(filters,
                kernel_size=num_conv,
                padding='same',
                activation='relu',
                strides=1)(conv_1)
flat = Flatten()(conv_2) # Since we are passing flat data anyway, we probably don't need this.
hidden = Dense(intermediate_dim, activation='relu')(flat)
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)



def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var ) * epsilon # the original VAE divides z_log_var with two -- why?

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])




encoder = Model(x, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')(latent_inputs)
decoder_mean = Dense(original_dim, activation='sigmoid')(decoder_h)

# h_decoded = decoder_h(z)
# x_decoded_mean = decoder_mean(h_decoded)

# decoder = Model(latent_inputs, x_decoded_mean, name='decoder')
decoder = Model(latent_inputs, decoder_mean, name='decoder')
decoder.summary()

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) # Double check wtf this is supposed to be
    return xent_loss + kl_loss


outputs = decoder(encoder(x)[2])
vae = Model(x, outputs, name='vae')

# vae = Model(x, x_decoded_mean)
# vae = Model(x, decoder_mean)

vae.compile(optimizer='adam', loss=vae_loss) # 'rmsprop'
vae.summary()





# N = 1000
# epochs = 2
# batch_size = int(N/10)
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)


###############

vae.save_weights('conv1dvae_cnn_galsim_cosmos.h5')

# plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")
# Saving
# ----------------------------------------------------------------------------

x_train_encoded = encoder.predict(x_train)

x_train_encoded = K.cast_to_floatx(x_train_encoded)

x_train_decoded = decoder.predict(x_train_encoded[0])

x_test_encoded = encoder.predict(x_test)

x_test_encoded = K.cast_to_floatx(x_test_encoded[0])

x_test_decoded = decoder.predict(x_test_encoded)

np.savetxt(DataDir+'conv1dvae_encoded_xtrain_cosmos.txt', x_train_encoded[0])
np.savetxt(DataDir+'conv1dvae_encoded_xtest_cosmos.txt', x_test_encoded[0])

# np.savetxt(DataDir+'cvae_decoded_xtrain_cosmos.txt', np.reshape(x_train_decoded, (x_train_decoded.shape[0], x_train_decoded.shape[1]*x_train_decoded.shape[2])))
# np.savetxt(DataDir+'cvae_decoded_xtest_cosmos.txt', np.reshape(x_test_decoded, (x_test_decoded.shape[0], x_test_decoded.shape[1]*x_test_decoded.shape[2])))


np.savetxt(DataDir+'conv1dvae_decoded_xtrain_cosmos.txt', x_train_decoded)
np.savetxt(DataDir+'conv1dvae_decoded_xtest_cosmos.txt', x_test_decoded)

# ---------------------- GP fitting -------------------------------


def gp_fit(weights, y_train):
    """
    Learns the GP related to the weigths matrix
    Input :
    - weights : From encoder (2-D) : x_train_encoded
    - y_train : Physical parameters to interpolate

    Output :
    - model : GP model
    """
    # Set the kernel
    # kernel = GPy.kern.Matern52(input_dim=params.shape[1], variance=.1, lengthscale=.1)
    kernel = GPy.kern.Matern52(input_dim=y_train.shape[1])

    # GP Regression
    model = GPy.models.GPRegression(y_train, weights, kernel=kernel)
    model.optimize()

    # Save model
    model.save_model(DataDir + 'gpfit_cvae', compress=True, save_data=True)
    return model


def gp_predict(model, params):
    """
    Predicts the weights matrix to feed inverse PCA from physical parameters.

    Input :
    - model : GP model
    - params : physical parameters (flux, radius, shear profile, psf fwhm)

    Output :
    - predic[0] : predicted weights
    """
    predic = model.predict(params)
    return predic[0]


gpmodel = gp_fit(x_train_encoded[0], y_train)

x_test_encoded = gp_predict(gpmodel, y_test)

np.savetxt(DataDir + 'x_test_encoded_64_5.txt', x_test_encoded)

x_test_decoded = decoder.predict(x_test_encoded)


# -------------------- Plotting routines --------------------------
#
# plt.figure()
# for i in range(10):
#     plt.subplot(3, 10, i+1)
#     plt.imshow(np.reshape(x_train[i], (image_size, image_size)))
#     # plt.title('Emulated image using PCA + GP '+str(i))
#     # plt.colorbar()
#     plt.subplot(3, 10, 10+i+1)
#     plt.imshow(np.reshape(x_train_decoded[i], (image_size, image_size)))
#     # plt.title('Simulated image using GalSim '+str(i))
#     # plt.colorbar()
#     plt.subplot(3, 10, 20+i+1)
#     plt.imshow(np.reshape(abs(x_train_decoded[i]-x_train[i]), (image_size, image_size)))
#
# plt.figure()
# for i in range(10):
#     plt.subplot(3, 10, i+1)
#     plt.imshow(np.reshape(x_test[i], (image_size, image_size)))
#     # plt.title('Emulated image using PCA + GP '+str(i))
#     # plt.colorbar()
#     plt.subplot(3, 10, 10+i+1)
#     plt.imshow(np.reshape(x_test_decoded[i], (image_size, image_size)))
#     # plt.title('Simulated image using GalSim '+str(i))
#     # plt.colorbar()
#     plt.subplot(3, 10, 20+i+1)
#     plt.imshow(np.reshape(abs(x_test_decoded[i]-x_test[i]), (image_size, image_size)))
#
# plt.show()

PlotScatter = True
if PlotScatter:

    w1 = 0
    w2 = 2
    # display a 2D plot of latent space (just 2 dimensions)
    plt.figure(figsize=(6, 6))

    x_train_encoded = encoder.predict(x_train)
    plt.scatter(x_train_encoded[0][:, w1], x_train_encoded[0][:, w2], c=y_train[:, 0], cmap='spring')
    plt.colorbar()

    x_test_encoded = encoder.predict(x_test)
    plt.scatter(x_test_encoded[0][:, w1], x_test_encoded[0][:, w2], c=y_test[:, 0], cmap='copper')
    plt.colorbar()
    # plt.title(fileOut)
    plt.savefig('cvae_Scatter_z'+'.png')

    # Plot losses
    n_epochs = np.arange(1, num_epochs+1)
    train_loss = vae.history.history['loss']
    val_loss = np.ones_like(train_loss)
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
    ax.plot(n_epochs, train_loss, '-', lw=1.5)
    ax.plot(n_epochs, val_loss, '-', lw=1.5)
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    ax.legend(['train loss', 'val loss'])
    plt.tight_layout()

plt.show()


PlotSample = True
if PlotSample:
    for i in range(10):
        plt.figure(91, figsize=(8,6))
        # plt.plot(ls, 10**(normFactor*x_train_decoded[i])/10**(normFactor*x_train[i]), 'r-', alpha = 0.8)
        # plt.plot(ls, 10**(normFactor*x_test_decoded[i])/10**(normFactor*x_test[i]), 'k-', alpha = 0.8)

        plt.plot(ls, x_train_decoded[i]/x_train[i][0,:], 'r-', alpha = 0.8)
        # plt.plot(ls, x_test_decoded[i]/x_test[i][0,:], 'k-', alpha = 0.8)

        # plt.ylim(0.85, 1.15)

        # plt.xscale('log')
        # plt.yscale('log')
        plt.ylabel('reconstructed/real')
        plt.title('train(red) and test (black)')
        plt.savefig(PlotsDir + 'Ratio_ttP'+str(num_para)+ClID+fileOut+'.png')


        if (i%2 == 1):
            plt.figure(654, figsize=(8,6))
            # plt.plot(ls, 10**(normFactor*x_test_decoded[i]), 'r-', alpha = 0.8)
            # plt.plot(ls, 10**(normFactor*x_test[i]), 'b--', alpha = 0.8)

            # plt.plot(ls, ( normFactor*x_test_decoded[i] ) + meanFactor, 'r-', alpha = 0.8)
            # plt.plot(ls, ( normFactor*x_test[i] ) + meanFactor, 'b--', alpha = 0.8)

            plt.plot(ls, ( normFactor*x_train_decoded[i] ) + meanFactor, 'r-', alpha = 0.8)
            plt.plot(ls, ( normFactor*x_train[i][0] ) + meanFactor, 'b--', alpha = 0.8)



            # plt.xscale('log')
            # plt.yscale('log')
            plt.title('Testing: reconstructed (red) and real (blue)')
            plt.savefig(PlotsDir + 'decoderTestP'+str(num_para)+ClID+ fileOut + '.png')

    plt.show()
