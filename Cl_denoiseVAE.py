"""

this script gave NaN loss  -- dunno where
Followed from https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

"""
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

original_dim = 2549#/2 +1  #2551 # mnist ~ 784
intermediate_dim2 = 1024#/2 #
intermediate_dim1 = 512#/2 #
intermediate_dim = 256#/2 #
latent_dim = 10

totalFiles = 512
TestFiles = 32 #128

batch_size = 4
num_epochs = 50 #110 #50
epsilon_mean = 1.0 # 1.0
epsilon_std = 1.0 # 1.0
learning_rate = 1e-5
decay_rate = 0.01

noise_factor = 0.00 # 0.0 necessary

# ----------------------------------------------------------------------------

# Q(z|X) -- encoder
inputs = Input(shape=(original_dim,))
h_q2 = Dense(intermediate_dim1, activation='relu')(inputs) # ADDED intermediate layer
h_q1 = Dense(intermediate_dim1, activation='relu')(h_q2) # ADDED intermediate layer
h_q = Dense(intermediate_dim, activation='relu')(h_q1)
mu = Dense(latent_dim, activation='linear')(h_q)
log_sigma = Dense(latent_dim, activation='linear')(h_q)

# ----------------------------------------------------------------------------

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(batch_size, latent_dim), mean=epsilon_mean, stddev=epsilon_std)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])

# ----------------------------------------------------------------------------

# P(X|z) -- decoder
decoder_hidden = Dense(latent_dim, activation='relu')
decoder_hidden1 = Dense(intermediate_dim, activation='relu') # ADDED intermediate layer
decoder_hidden2 = Dense(intermediate_dim1, activation='relu') # ADDED intermediate layer
decoder_hidden3 = Dense(intermediate_dim2, activation='relu') # ADDED intermediate layer
decoder_out = Dense(original_dim, activation='sigmoid')

h_p1 = decoder_hidden(z)
h_p2 = decoder_hidden1(h_p1) # ADDED intermediate layer
h_p3 = decoder_hidden2(h_p2) # ADDED intermediate layer
h_p4 = decoder_hidden3(h_p3) # ADDED intermediate layer
outputs = decoder_out(h_p4)

# ----------------------------------------------------------------------------


# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu)

# Generator model, generate new data given latent variable z
# d_in = Input(shape=(latent_dim,))
# d_h = decoder_hidden(d_in)
# d_h1 = decoder_hidden1(d_h)
# d_h2 = decoder_hidden2(d_h1)
# d_out = decoder_out(d_h2)
# decoder = Model(d_in, d_out)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))

_h_decoded = decoder_hidden(decoder_input)
_h0_decoded = decoder_hidden1(_h_decoded)    ## ADDED layer_1
_h1_decoded = decoder_hidden2(_h0_decoded)    ## ADDED --- should replicate decoder arch
_h2_decoded = decoder_hidden3(_h1_decoded)    ## ADDED --- should replicate decoder arch
_x_decoded_mean = decoder_out(_h2_decoded)
decoder = Model(decoder_input, _x_decoded_mean)


# -------------------------------------------------------------
#CUSTOM LOSS

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # recon = K.categorical_crossentropy(y_pred, y_true)

    # recon = losses.mean_squared_error(y_pred, y_true)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5*K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl

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

x_train = x_train[:,2:] #
x_test =  x_test[:,2:] #

# x_train = x_train[:,2::2] #
# x_test =  x_test[:,2::2] #

print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')


# meanFactor = np.min( [np.min(x_train), np.min(x_test ) ])
# print('-------mean factor:', meanFactor)
# x_train = x_train.astype('float32') - meanFactor #/ 255.
# x_test = x_test.astype('float32') - meanFactor #/ 255.
# np.save('../Cl_data/Data/meanfactor_'+str(totalFiles)+'.npy', meanFactor)
#

# x_train = np.log10(x_train[:,::2]) #x_train[:,2:] #
# x_test =  np.log10(x_test[:,::2]) #x_test[:,2:] #

normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
print('-------normalization factor:', normFactor)
x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.
np.save('../Cl_data/Data/normfactor_'+str(totalFiles)+'.npy', normFactor)


x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


## Trying to get x_train ~ (-1, 1) -- doesn't work well
# x_mean = np.mean(x_train, axis = 0)
# x_train = x_train - x_mean
# x_test = x_test - x_mean


## ADD noise
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


# plt.plot(x_train_noisy.T)
# ------------------------------------------------------------------------------

#TRAIN       -- NaN losses Uhhh
adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                          decay=decay_rate)


vae.compile(optimizer='adam', loss=vae_loss)

# vae.optimizer.lr.set_value(learning_rate)

K.set_value(vae.optimizer.lr, learning_rate)
K.set_value(vae.optimizer.decay, decay_rate)


vae.fit(x_train_noisy, x_train, shuffle=True, batch_size=batch_size, nb_epoch=num_epochs, verbose=2,
        validation_data=(x_test_noisy, x_test))

print('--------learning rate : ', K.eval(vae.optimizer.lr) )
# ----------------------------------------------------------------------------

x_train_encoded = encoder.predict(x_train)
x_decoded = decoder.predict(x_train_encoded)

np.save('../Cl_data/Data/encoded_xtrain_'+str(totalFiles)+'.npy', x_train_encoded)

# -------------------- Save model/weights --------------------------


SaveModel = True
if SaveModel:
    epochs = np.arange(1, num_epochs+1)
    train_loss = vae.history.history['loss']
    val_loss = vae.history.history['val_loss']

    training_hist = np.vstack([epochs, train_loss, val_loss])

    # fileOut = 'Stack_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)

    fileOut = 'DenoiseModel_tot'+str(totalFiles)+'_batch'+str(batch_size)+'_lr'+str(
        learning_rate)+'_decay'+str(decay_rate)+'_z'+str(latent_dim)+'_epoch'+str(num_epochs)

    vae.save('../Cl_data/Model/fullAE_' + fileOut + '.hdf5')
    encoder.save('../Cl_data/Model/Encoder_' + fileOut + '.hdf5')
    decoder.save('../Cl_data/Model/Decoder_' + fileOut + '.hdf5')
    np.save('../Cl_data/Model/TrainingHistory_'+fileOut+'.npy', training_hist)

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
    plt.show()


# ls = np.log10(np.load('../Cl_data/Data/ls_'+str(totalFiles)+'.npy')[2::2])
ls = np.load('../Cl_data/Data/ls_'+str(totalFiles)+'.npy')[2:]

PlotSample = True
if PlotSample:
    for i in range(3,10):
        plt.figure(91, figsize=(8,6))
        plt.plot(ls, x_decoded[i], 'r--', alpha = 0.8)
        plt.plot(ls, x_train[i], 'b--', alpha = 0.8)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.title('reconstructed - red')

    plt.show()


plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    epochs = np.arange(1, num_epochs+1)
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
    # plt.savefig('../Cl_data/Plots/Training_loss.png')


PlotModel = False
if PlotModel:
    from keras.utils.vis_utils import plot_model
    fileOut = '../Cl_data/Plots/ArchitectureFullAE.png'
    plot_model(vae, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = '../Cl_data/Plots/ArchitectureEncoder.png'
    plot_model(encoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = '../Cl_data/Plots/ArchitectureDecoder.png'
    plot_model(decoder, to_file=fileOut, show_shapes=True, show_layer_names=True)
