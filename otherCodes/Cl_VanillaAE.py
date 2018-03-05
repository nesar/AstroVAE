"""

Followed from https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

"""
import numpy as np
np.random.seed(1337) # for reproducibility


from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import optimizers
from keras import losses

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import keras.backend as K

import params
import Cl_load
# import SetPub
# SetPub.set_pub()

###################### PARAMETERS ##############################

original_dim = params.original_dim # 2549
intermediate_dim2 = params.intermediate_dim2 # 1024
intermediate_dim1 = params.intermediate_dim1 # 512
intermediate_dim = params.intermediate_dim # 256
latent_dim = params.latent_dim # 10

totalFiles = params.totalFiles # 512
TestFiles = params.TestFiles # 32

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


################# ARCHITECTURE ###############################

input_img = Input(shape=(original_dim,))
encoded = Dense(intermediate_dim2, activation='relu')(input_img)
encoded = Dense(intermediate_dim1, activation='relu')(encoded)
encoded = Dense(intermediate_dim, activation='relu')(encoded)
encoded = Dense(latent_dim, activation='relu')(encoded)

decoded = Dense(intermediate_dim, activation='relu')(encoded)
decoded = Dense(intermediate_dim1, activation='relu')(decoded)
decoded = Dense(intermediate_dim2, activation='sigmoid')(decoded)
decoded = Dense(original_dim, activation='sigmoid')(decoded)


ae = Model(input_img, decoded)

ae.compile(optimizer='adadelta', loss='binary_crossentropy')

K.set_value(ae.optimizer.lr, learning_rate)
K.set_value(ae.optimizer.decay, decay_rate)


# create the encoder model
encoder = Model(inputs=input_img, outputs=encoded)

# create a placeholder for an encoded (32-dimensional) input
latent_input = Input(shape=(latent_dim,))
# retrieve the last layer of the autoencoder model
# decoder_layer = ae.layers[-1]

deco = ae.layers[-4](latent_input)
deco = ae.layers[-3](deco)
deco = ae.layers[-2](deco)
deco = ae.layers[-1](deco)
# create the decoder model
decoder = Model(latent_input, deco)

# create the decoder model
decoder = Model(inputs=latent_input, outputs=decoder(latent_input))


# ----------------------------- i/o ------------------------------------------

train_path = DataDir+'LatinCl_'+str(totalFiles)+'.npy'
train_target_path =  DataDir+'LatinPara5_'+str(totalFiles)+'.npy'
test_path = DataDir+'LatinCl_'+str(TestFiles)+'.npy'
test_target_path =  DataDir+'LatinPara5_'+str(TestFiles)+'.npy'

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
# np.save(DataDir+'meanfactor_'+str(totalFiles)+'.npy', meanFactor)
#

# x_train = np.log10(x_train[:,::2]) #x_train[:,2:] #
# x_test =  np.log10(x_test[:,::2]) #x_test[:,2:] #

normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
print('-------normalization factor:', normFactor)
x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.
np.save(DataDir+'normfactor_'+str(totalFiles)+'.npy', normFactor)


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

#TRAIN

ae.fit(x_train_noisy, x_train,
                epochs=num_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test_noisy, x_test), verbose=2)

print('--------learning rate : ', K.eval(ae.optimizer.lr) )



# ----------------------------------------------------------------------------

x_train_encoded = encoder.predict(x_train)
x_train_decoded = decoder.predict(x_train_encoded)

x_test_encoded = encoder.predict(x_test)
x_test_decoded = decoder.predict(x_test_encoded)

np.save(DataDir+'encoded_xtrain_'+str(totalFiles)+'.npy', x_train_encoded)

# -------------------- Save model/weights --------------------------


SaveModel = True
if SaveModel:
    epochs = np.arange(1, num_epochs+1)
    train_loss = ae.history.history['loss']
    val_loss = ae.history.history['val_loss']

    training_hist = np.vstack([epochs, train_loss, val_loss])


    ae.save(ModelDir+'fullAE_' + fileOut + '.hdf5')
    encoder.save(ModelDir + 'Encoder_' + fileOut + '.hdf5')
    decoder.save(ModelDir + 'Decoder_' + fileOut + '.hdf5')
    np.save(ModelDir + 'TrainingHistory_'+fileOut+'.npy', training_hist)

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
    plt.savefig( PlotsDir + 'Scatter_z'+fileOut+'.png')
    plt.show()


# ls = np.log10(np.load(DataDir+'ls_'+str(totalFiles)+'.npy')[2::2])
ls = np.load(DataDir+'Latinls_'+str(totalFiles)+'.npy')[2:]


PlotSample = True
if PlotSample:
    for i in range(10):
        plt.figure(91, figsize=(8,6))
        plt.plot(ls, x_train_decoded[i]/x_train[i], 'b-', alpha = 0.8)
        plt.plot(ls, x_test_decoded[i]/x_test[i], 'k-', alpha = 0.8)
        # plt.plot(ls, x_train[i], 'b--', alpha = 0.8)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.title('reconstructed/real')
        plt.savefig(PlotsDir + 'Ratio_tt'+fileOut+'.png')

        if (i%3 == 1):
            plt.figure(654, figsize=(8,6))
            plt.plot(ls, x_test_decoded[i], 'r-', alpha = 0.8)
            plt.plot(ls, x_test[i], 'b--', alpha = 0.8)
            # plt.plot(ls, x_train[i], 'b--', alpha = 0.8)
            # plt.xscale('log')
            # plt.yscale('log')
            plt.title('reconstructed (red) and real (blue)')
            plt.savefig(PlotsDir + 'decoderTest' + fileOut + '.png')

            #plt.show()


plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    epochs = np.arange(1, num_epochs+1)
    train_loss = ae.history.history['loss']
    val_loss = ae.history.history['val_loss']


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
    plt.savefig(PlotsDir+'Training_loss.png')


PlotModel = False
if PlotModel:
    from keras.utils.vis_utils import plot_model
    fileOut = PlotsDir + 'ArchitectureFullAE.png'
    plot_model(ae, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = PlotsDir + 'ArchitectureEncoder.png'
    plot_model(encoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = PlotsDir + 'ArchitectureDecoder.png'
    plot_model(decoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

plt.show()