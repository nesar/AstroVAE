"""

Variational Autoencoding for CMB power spectra.

Training scheme is unsupervised, i.e., the autoencoder is not given information about
cosmological parameters.



This script uses Keras for neural network architecture.

"""
import numpy as np


from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import optimizers
from keras import losses

import matplotlib as mpl
# mpl.use('Agg')

import matplotlib.pyplot as plt
import keras.backend as K

import params
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
#intermediate_dim3 = params.intermediate_dim3 # 1600
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


# ----------------------------- i/o ------------------------------------------

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

#----------------------------------------------------------------------------

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

################# ARCHITECTURE ###############################

from keras.optimizers import Adam
from keras.layers.merge import concatenate as concat
from keras.callbacks import EarlyStopping




m = 250 # batch size
n_z = 2 # latent space size
encoder_dim1 = 128 # dim of encoder hidden layer
decoder_dim = 128 # dim of decoder hidden layer
decoder_out_dim = params.original_dim # dim of decoder output layer
activ = 'relu'
optim = Adam(lr=0.001)


n_x = x_train.shape[1]
n_y = y_train.shape[1]


n_epoch = 50

X = Input(shape=(n_x,))
label = Input(shape=(n_y,))


inputs = concat([X, label])


encoder_h = Dense(encoder_dim1, activation=activ)(inputs)
mu = Dense(n_z, activation='linear')(encoder_h)
l_sigma = Dense(n_z, activation='linear')(encoder_h)


def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(l_sigma / 2) * eps


# Sampling latent space
z = Lambda(sample_z, output_shape = (n_z, ))([mu, l_sigma])





z = Lambda(sample_z, output_shape = (n_z, ))([mu, l_sigma])

# merge latent space with label
zc = concat([z, label])




decoder_hidden = Dense(decoder_dim, activation=activ)
decoder_out = Dense(decoder_out_dim, activation='sigmoid')
h_p = decoder_hidden(zc)
outputs = decoder_out(h_p)




def vae_loss(y_true, y_pred):
    recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)
    return recon + kl

def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

def recon_loss(y_true, y_pred):
	return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)





cvae = Model([X, label], outputs)
encoder = Model([X, label], mu)

d_in = Input(shape=(n_z+n_y,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)


cvae.compile(optimizer=optim, loss=vae_loss, metrics = [KL_loss, recon_loss])

print(cvae.summary())


# compile and fit
cvae_hist = cvae.fit([x_train, y_train], x_train, verbose = 1, batch_size=m, epochs=n_epoch,
							validation_data = ([x_test, y_test], x_test),
							callbacks = [EarlyStopping(patience = 5)])





# K.set_value(vae.optimizer.lr, learning_rate)
# K.set_value(vae.optimizer.decay, decay_rate)




# from keras.utils import plot_model
# plot_model(vae, show_shapes = True, show_layer_names = False,  to_file='model.png' )


#TRAIN

# ----------------------------------------------------------------------------

x_train_encoded = encoder.predict(x_train)
x_train_decoded = decoder.predict(x_train_encoded)

x_test_encoded = encoder.predict(x_test)
x_test_decoded = decoder.predict(x_test_encoded)

np.savetxt(DataDir+'encoded_xtrainP'+str(num_para)+ClID+'_'+ fileOut +'.txt', x_train_encoded)
np.savetxt(DataDir+'encoded_xtestP'+str(num_para)+ClID+'_'+ fileOut +'.txt', x_test_encoded)

# np.save(DataDir+'para5_'+str(num_train)+'.npy', y_train)
# -------------------- Save model/weights --------------------------


SaveModel = True
if SaveModel:
    epochs = np.arange(1, num_epochs+1)
    train_loss = cvae_hist.history.history['loss']
    val_loss = cvae_hist.history.history['val_loss']

    training_hist = np.vstack([epochs, train_loss, val_loss])


    cvae.save(ModelDir+'fullAEP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    encoder.save(ModelDir + 'EncoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    decoder.save(ModelDir + 'DecoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    np.savetxt(ModelDir + 'TrainingHistoryP'+str(num_para)+ClID+'_'+fileOut+'.txt', training_hist)

# -------------------- Plotting routines --------------------------
PlotScatter = True
if PlotScatter:
    # display a 2D plot of latent space (just 2 dimensions)
    plt.figure(figsize=(6, 6))

    x_train_encoded = encoder.predict(x_train)
    plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c=y_train[:, 0], cmap='spring')
    plt.colorbar()

    x_test_encoded = encoder.predict(x_test)
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test[:, 0], cmap='copper')
    plt.colorbar()
    plt.title(fileOut)
    plt.savefig( PlotsDir + 'Scatter_z'+ClID+fileOut+'.png')



PlotSample = True
if PlotSample:
    for i in range(10):
        plt.figure(91, figsize=(8,6))
        # plt.plot(ls, 10**(normFactor*x_train_decoded[i])/10**(normFactor*x_train[i]), 'r-', alpha = 0.8)
        # plt.plot(ls, 10**(normFactor*x_test_decoded[i])/10**(normFactor*x_test[i]), 'k-', alpha = 0.8)

        plt.plot(ls, x_train_decoded[i]/x_train[i], 'r-', alpha = 0.8)
        plt.plot(ls, x_test_decoded[i]/x_test[i], 'k-', alpha = 0.8)

        plt.ylim(0.85, 1.15)

        # plt.xscale('log')
        # plt.yscale('log')
        plt.ylabel('reconstructed/real')
        plt.title('train(red) and test (black)')
        plt.savefig(PlotsDir + 'Ratio_ttP'+str(num_para)+ClID+fileOut+'.png')


        if (i%2 == 1):
            plt.figure(654, figsize=(8,6))
            # plt.plot(ls, 10**(normFactor*x_test_decoded[i]), 'r-', alpha = 0.8)
            # plt.plot(ls, 10**(normFactor*x_test[i]), 'b--', alpha = 0.8)

            plt.plot(ls, ( normFactor*x_test_decoded[i] ) + meanFactor, 'r-', alpha = 0.8)
            plt.plot(ls, ( normFactor*x_test[i] ) + meanFactor, 'b--', alpha = 0.8)


            # plt.xscale('log')
            # plt.yscale('log')
            plt.title('Testing: reconstructed (red) and real (blue)')
            plt.savefig(PlotsDir + 'decoderTestP'+str(num_para)+ClID+ fileOut + '.png')

    plt.show()


print(fileOut)
print(ClID)
print('--------max ratio (train) : ', np.max(x_train_decoded/x_train) )
print('--------max ratio (test)  : ', np.max(x_test_decoded/x_test) )


plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    epochs = np.arange(1, num_epochs+1)
    train_loss = cvae_hist.history.history['loss']
    val_loss = cvae_hist.history.history['val_loss']


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
    # plt.savefig(PlotsDir+'Training_loss.png')


PlotModel = False
if PlotModel:
    from keras.utils.vis_utils import plot_model
    fileOut = PlotsDir + 'ArchitectureFullAE.png'
    plot_model(cvae, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = PlotsDir + 'ArchitectureEncoder.png'
    plot_model(encoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = PlotsDir + 'ArchitectureDecoder.png'
    plot_model(decoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

plt.show()
