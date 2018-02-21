"""
GP fit for W matrix or latent z (encoded representation)

Uses George - package by Dan Foreman McKay - better integration with his MCMC package.
pip install george  - http://dan.iel.fm/george/current/user/quickstart/

"""
print(__doc__)

# Higdon et al 2008, 2012
# Check David's talk for plots of spectrum, and other things.

import numpy as np

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model

import params
import Cl_load
#import SetPub
#SetPub.set_pub()



def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)



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


LoadModel = True
if LoadModel:

    # fileOut = 'VanillaModel_tot'+str(totalFiles)+'_batch'+str(batch_size)+'_lr'+str( learning_rate)+'_decay'+str(decay_rate)+'_z'+str(latent_dim)+'_epoch'+str(num_epochs)

    # vae = load_model(ModelDir + 'fullAE_' + fileOut + '.hdf5')
    encoder = load_model(ModelDir + 'EncoderP4_' + fileOut + '.hdf5')
    decoder = load_model(ModelDir + 'DecoderP4_' + fileOut + '.hdf5')
    history = np.load(ModelDir + 'TrainingHistoryP4_'+fileOut+'.npy')


import george
from george.kernels import Matern32Kernel #, ConstantKernel, WhiteKernel, Matern52Kernel

# kernel = ConstantKernel(0.5, ndim=5) * Matern52Kernel(0.9, ndim=5) + WhiteKernel(0.1, ndim=5)
kernel = Matern32Kernel(0.5, ndim=4)

# ----------------------------- i/o ------------------------------------------


train_path = DataDir + 'LatinClP4_'+str(totalFiles)+'.npy'
train_target_path =  DataDir + 'LatinPara5P4_'+str(totalFiles)+'.npy'
test_path = DataDir + 'LatinClP4_'+str(TestFiles)+'.npy'
test_target_path =  DataDir + 'LatinPara5P4_'+str(TestFiles)+'.npy'

camb_in = Cl_load.cmb_profile(train_path = train_path,  train_target_path = train_target_path , test_path = test_path, test_target_path = test_target_path, num_para=5)


(x_train, y_train), (x_test, y_test) = camb_in.load_data()


x_train = x_train[:,2:]
x_test = x_test[:,2:]

# x_train = x_train[:,2::2]
# x_test = x_test[:,2::2]

print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')


# meanFactor = np.min( [np.min(x_train), np.min(x_test ) ])
# print('-------mean factor:', meanFactor)
# x_train = x_train.astype('float32') - meanFactor #/ 255.
# x_test = x_test.astype('float32') - meanFactor #/ 255.
#

# x_train = np.log10(x_train) #x_train[:,2:] #
# x_test =  np.log10(x_test) #x_test[:,2:] #

normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
print('-------normalization factor:', normFactor)

x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# ------------------------------------------------------------------------------
# PLOTTING y_train and y_test

# import pandas as pd
# plt.figure(431)
# AllLabels = [r'$\Omega_m$', r'$\Omega_b$', r'$\sigma_8$', r'$h$', r'$n_s$']
# df = pd.DataFrame(y_train, columns=AllLabels)
# axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'b')
# df = pd.DataFrame(y_test, columns=AllLabels)
# axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'k')
# plt.tight_layout()
# plt.savefig('scatter_matrix.png')

# ------------------------------------------------------------------------------

# y_train1 = np.load(DataDir+'para5_'+str(totalFiles)+'.npy')


X1 = y_train[:, 0][:, np.newaxis]
X1a = rescale01(np.min(X1), np.max(X1), X1)

X2 = y_train[:, 1][:, np.newaxis]
X2a = rescale01(np.min(X2), np.max(X2), X2)

X3 = y_train[:, 2][:, np.newaxis]
X3a = rescale01(np.min(X3), np.max(X3), X3)

X4 = y_train[:, 3][:, np.newaxis]
X4a = rescale01(np.min(X4), np.max(X4), X4)

# X5 = y_train[:, 4][:, np.newaxis]
# X5a = rescale01(np.min(X5), np.max(X5), X5)


XY = np.array(np.array([X1a, X2a, X3a, X4a])[:, :, 0])[:, np.newaxis]


# # ------------------------------------------------------------------------------
y = np.load(DataDir + 'encoded_xtrainP4_'+str(totalFiles)+'.npy').T
encoded_xtest_original = np.load(DataDir+'encoded_xtestP4_'+str(totalFiles)+'.npy')


# ymax = y.max()
# y = y/ymax

# ------------------------------------------------------------------------------
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# ------------------------------------------------------------------------------

# PlotSampleID = [6, 4, 23, 26, 17, 12, 30, 4]
PlotSampleID = [0, 1, 2,  12, 0, 4, 7]

max_relError = 0
ErrTh = 5
PlotRatio = True

W_predArray = np.zeros(shape=(TestFiles,latent_dim))


if PlotRatio:
    # ls = np.log10(np.load('../Cl_data/Data/Latinls_' + str(TestFiles) + '.npy')[2:])
    # PkOriginal = np.log10(np.load('../Cl_data/Data/LatinCl_'+str(TestFiles)+'.npy')[:,
    # 2:]) # Original

    ls = np.load(DataDir + 'LatinlsP4_' + str(TestFiles) + '.npy')[2:]#[2::2]

    # Cl_Original = np.load(DataDir + 'LatinCl_'+str(TestFiles)+'.npy')[:,2:]
    # RealParaArray = np.load(DataDir + 'LatinPara5_'+str(TestFiles)+'.npy')

    Cl_Original = (normFactor*x_test)#[2:3]
    RealParaArray = y_test#[2:3]


    # Cl_Original = (normFactor*x_train)[0:10]
    # RealParaArray = y_train[0:10]



    for i in range(np.shape(RealParaArray)[0]):

        RealPara = RealParaArray[i]

        RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
        RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
        RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2])
        RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
        # RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])

        test_pts = RealPara[:4].reshape(4, -1).T

        # ------------------------------------------------------------------------------
        # y = np.load('../Pk_data/SVDvsVAE/encoded_xtrain.npy').T

        W_pred = np.array([np.zeros(shape=latent_dim)])
        gp = {}
        for j in range(latent_dim):
            gp["fit{0}".format(j)] = george.GP(kernel)
            gp["fit{0}".format(j)].compute(XY[:, 0, :].T)
            W_pred[:, j] = gp["fit{0}".format(j)].predict(y[j], test_pts)[0]

        # ------------------------------------------------------------------------------

        W_predArray[i] = W_pred
        # x_decoded = decoder.predict(W_pred*ymax)# + meanFactor
        x_decoded = decoder.predict(W_pred)# + meanFactor


        plt.figure(94, figsize=(8,6))
        plt.title('Autoencoder+GP fit')
        # cl_ratio = 10**(normFactor*x_decoded[0])/10**(Cl_Original[i])

        cl_ratio = (normFactor*x_decoded[0])/(Cl_Original[i])


        relError = 100*np.abs(cl_ratio - 1)

        plt.plot(ls, cl_ratio, alpha=.8, lw = 1.0)

        # plt.xscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$C_l^{GPAE}$/$C_l^{Original}$')
        plt.title(fileOut)
        # plt.legend()
        plt.tight_layout()



        if i in PlotSampleID:

            plt.figure(99, figsize=(8,6))
            plt.title('Autoencoder+GP fit')
            # plt.plot(ls, normFactor * x_test[::].T, 'gray', alpha=0.1)

            # plt.plot(ls, 10**(normFactor*x_decoded[0]), 'r--', alpha= 0.5, lw = 1, label = 'emulated')
            # plt.plot(ls, 10**(Cl_Original[i]), 'b--', alpha=0.5, lw = 1, label = 'original')

            plt.plot(ls, (normFactor*x_decoded[0]), 'r--', alpha= 0.5, lw = 1, label = 'emulated')
            plt.plot(ls, (Cl_Original[i]), 'b--', alpha=0.5, lw = 1, label = 'original')


            # plt.xscale('log')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$C_l$')
            plt.legend()
            # plt.tight_layout()

            plt.plot(ls[relError > ErrTh], normFactor*x_decoded[0][relError > ErrTh], 'gx', alpha=0.8, label='bad eggs', markersize = '1')
            plt.title(fileOut)

            plt.savefig(PlotsDir + 'TestP4'+fileOut+'.png')
        #plt.show()
        print(i, 'ERR0R min max (per cent):', np.array([(relError).min(), (relError).max()]) )

        max_relError = np.max( [np.max(relError) , max_relError] )

    plt.figure(94, figsize=(8,6))
    plt.axhline(y=1, ls='-.', lw=1.5)
    plt.savefig(PlotsDir + 'RatioP4'+fileOut+'.png')

    #plt.show()
print(50*'-')
print('file:', fileOut)
# ------------------------------------------------------------------------------


plotLoss = True
if plotLoss:

    epochs =  history[0,:]
    train_loss = history[1,:]
    val_loss = history[2,:]


    plt.figure(867)
    fig, ax = plt.subplots(1,1, sharex= True, figsize = (8,6))
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epochs,train_loss, '-', lw =1.5)
    ax.plot(epochs,val_loss, '-', lw = 1.5)
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train loss','val loss'])
    #plt.text(5.75, 0.15, 'MaxRelError: %d'%np.int(max_relError) , fontsize=15)
    plt.title(fileOut)
    plt.tight_layout()
    plt.savefig(PlotsDir + 'TrainingLoss_'+fileOut+'_relError'+ str( np.int(max_relError) ) +'.png')

plt.show()

print(50*'#')
print(fileOut)
print
print('max rel error:', str( (max_relError) ) )
print(50*'#')




# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


## Sent to Mickeal  #---------------------------------------------


(x_train1, y_train1), (x_test1, y_test1) = camb_in.load_data()

# para5_train, encoded_train

para5_train = y_train1
encoded_train = np.load(DataDir + 'encoded_xtrainP4_'+str(totalFiles)+'.npy')


# para5_new, encoded_testing_new ( to check encoded_test_original

para5_new = y_test1
encoded_test_original = np.load(DataDir+'encoded_xtestP4_'+str(totalFiles)+'.npy')



np.savetxt('para5_trainP4.txt', para5_train)
np.savetxt('para5_newP4.txt', para5_new)

np.savetxt('encoded_trainP4.txt', encoded_train)
np.savetxt('encoded_test_originalP4.txt', encoded_test_original)


plt.figure(1542)
# plt.plot(encoded_train.T, 'b', alpha = 0.02 )
plt.plot(encoded_test_original.T, 'o', alpha = 0.2)
plt.plot(W_predArray.T, 'ko', alpha = 0.3)
plt.show()


plt.figure(5343)
plt.plot(W_predArray/encoded_xtest_original, 'ko', markersize=2)
plt.yscale('symlog')


# plt.figure(5343)
indCheck = 6
plt.scatter(W_predArray[:,indCheck], encoded_xtest_original[:,indCheck], c = para5_new[:,0])



plt.figure(431)
import pandas as pd

AllLabels = []

for ind in np.arange(1, 4+1):
    AllLabels.append(str("v{0}".format(ind)))

for ind in np.arange(1, latent_dim+1):
    AllLabels.append(str("z{0}".format(ind)))


inputArray = np.hstack([y_train, encoded_train])
df = pd.DataFrame(inputArray, columns=AllLabels)
axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'b')


# AllLabels = [r'$\Omega_m$', r'$\Omega_b$', r'$\sigma_8$', r'$h$', r'$n_s$']
# df = pd.DataFrame(encoded_test_original, columns=AllLabels)
# axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'b')
# df = pd.DataFrame(  W_predArray, columns=AllLabels)
# axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'k')
plt.show()












# ------------------------------------------------------------------------------

# Cosmic Variance emulator


# Next step -- uncertainty quantification -- check Laurence et al
# Parameter estimation  -- mcmc or variational bayes
# Go back to check P(k) -- we may just use the simulation suite data.
# Generate better data sampling - latin hc?
#
# Deeper network!
# adding hidden layers seems to be fixed -- check again

# Talk to Jason
# CAMB install gfortran issue in Phoenix
# Generate lots of outputs
# training check using 32 files.
# Why emulate when there's a theoretical model?

# Still haven't figured how to change sigma_8 in CAMB

# CAMB - linear theory -- good for demonstration
# Future -- use simulations for a proper cmb power spectrum

# Change loss function and see if it changes anything
## can loss include something from GP?

# combine a few files -- one for io, one for training (AE+GP), one for testing?

## encoded arrays are all same!!!

## Tried avg, didn't help
## Try fft ??


## Plot GP functions cuz people want to look at it

## Hyper-parameter optimization softwares

#plt.figure(35)
#for i in range(10):
#    plt.plot(np.fft.rfft(x_train[i,:]))
#plt.xscale('log')

#plt.plot( np.fft.irfft( np.fft.rfft(x_train[0,:]) ), 'r' )
#plt.plot(x_train[0,:], 'b-.')



################3333333

# GP Doesn't seem to interpolate z space
# Even training points aren't mapped properly!!!
# Is the z space too weird?

# Same problem with my PCA+GP analysis
# Is the problems with epison_std being too small??
