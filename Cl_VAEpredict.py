"""
GP fit for W matrix - with only 2 eigenvalues

Uses George - package by Dan Foreman McKay - better integration with his MCMC package.
pip install george  - http://dan.iel.fm/george/current/user/quickstart/

"""
print(__doc__)

# Higdon et al 2008, 2012
# Check David's talk for plots of spectrum, and other things.

import numpy as np



from matplotlib import pyplot as plt

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
# ExpSineSquared, DotProduct,
# ConstantKernel)

import george
import SetPub
SetPub.set_pub()



def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)



totalFiles = 100
TestFiles = 20

latent_dim = 5


# length_scaleParameter = 1.0
# length_scaleBoundMin = 0.1
# length_scaleBoundMax = 0.3

# kernels = [1.0 * Matern(length_scale=length_scaleParameter, length_scale_bounds=(length_scaleBoundMin, length_scaleBoundMax),
# nu=1.5)]

# from george import kernels

# k1 = 66.0**2 * kernels.ExpSquaredKernel(67.0**2)
# k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(2.0 / 1.3**2, 1.0)
# k3 = 0.66**2 * kernels.RationalQuadraticKernel(0.78, 1.2**2)
# k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2) + kernels.WhiteKernel(0.19)
# kernel = k1 + k2 + k3 + k4
# kernel = k1

from george.kernels import Matern32Kernel, ConstantKernel, WhiteKernel

# kernel = ConstantKernel(0.5, ndim=5) * Matern32Kernel(0.5, ndim=5) + WhiteKernel(0.1, ndim=5)
kernel = Matern32Kernel(0.5, ndim=5)


# ------------------------------------------------------------------------------

# hmf = np.loadtxt('Data/HMF_5Para.txt')
# hmf = np.load('../Pk_data/Para5.npy')

# Load from pk load instead
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

normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
print('-------normalization factor:', normFactor)

x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# ------------------------------------------------------------------------------

hmf = y_train

# ------------------------------------------------------------------------------



X1 = hmf[:, 0][:, np.newaxis]
X1a = rescale01(np.min(X1), np.max(X1), X1)

X2 = hmf[:, 1][:, np.newaxis]
X2a = rescale01(np.min(X2), np.max(X2), X2)

X3 = hmf[:, 2][:, np.newaxis]
X3a = rescale01(np.min(X3), np.max(X3), X3)

X4 = hmf[:, 3][:, np.newaxis]
X4a = rescale01(np.min(X4), np.max(X4), X4)

X5 = hmf[:, 4][:, np.newaxis]
X5a = rescale01(np.min(X5), np.max(X5), X5)


XY = np.array(np.array([X1a, X2a, X3a, X4a, X5a])[:, :, 0])[:, np.newaxis]


# ------------------------------------------------------------------------------
# Test Sample
# This part will go inside likelihood -- Anirban
# RealPara = np.load('../Cl_data/Data/LatinPara5_2.npy')[0]
# RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
# RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
# # RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2]) ## ns = 0.8 constant while training
# RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
# RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])
#
# test_pts = RealPara.reshape(5, -1).T
#
# # ------------------------------------------------------------------------------
y = np.load('../Cl_data/Data/encoded_xtrain_'+str(totalFiles)+'.npy').T
#
# W_pred = np.array([np.zeros(shape=latent_dim)])
# gp={}
# for i in range(latent_dim):
#     gp["fit{0}".format(i)]= george.GP(kernel)
#     gp["fit{0}".format(i)].compute(XY[:, 0, :].T)
#     W_pred[:,i] = gp["fit{0}".format(i)].predict(y[i], test_pts)[0]
#
# # ------------------------------------------------------------------------------
# Decoder acts here
# Have to save and load model architecture, and weights

from keras.models import load_model

fileOut = 'Model_'+str(totalFiles)
# vae = load_model('../Pk_data/fullAE_' + fileOut + '.hdf5')
encoder = load_model('../Cl_data/Model/Encoder_' + fileOut + '.hdf5')
decoder = load_model('../Cl_data/Model/Decoder_' + fileOut + '.hdf5')
history = np.load('../Cl_data/Model/TrainingHistory_'+fileOut+'.npy')

# generator = Model(decoder_input, _x_decoded_mean)
# x_decoded = generator.predict(W_pred.T[0,:,:])
# x_decoded = decoder.predict(W_pred)



plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    epochs =  history[0,:]
    train_loss = history[1,:]
    val_loss = history[2,:]


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



# ------------------------------------------------------------------------------
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# ------------------------------------------------------------------------------

PlotRatio = True
if PlotRatio:
    totalFiles = 20
    ls = np.load('../Cl_data/Data/Latinls_' + str(totalFiles) + '.npy')[2:]
    PkOriginal = np.load('../Cl_data/Data/LatinCl_'+str(totalFiles)+'.npy')[:,2:] # Original
    RealParaArray = np.load('../Cl_data/Data/LatinPara5_'+str(totalFiles)+'.npy')

    for i in range(np.shape(RealParaArray)[0]):

        RealPara = RealParaArray[i]

        RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
        RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
        RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2])
        RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
        RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])

        test_pts = RealPara[:5].reshape(5, -1).T

        # ------------------------------------------------------------------------------
        # y = np.load('../Pk_data/SVDvsVAE/encoded_xtrain.npy').T

        W_pred = np.array([np.zeros(shape=latent_dim)])
        gp = {}
        for j in range(latent_dim):
            gp["fit{0}".format(j)] = george.GP(kernel)
            gp["fit{0}".format(j)].compute(XY[:, 0, :].T)
            W_pred[:, j] = gp["fit{0}".format(j)].predict(y[j], test_pts)[0]

        # ------------------------------------------------------------------------------

        x_decoded = decoder.predict(W_pred)


        plt.figure(94, figsize=(8,6))
        plt.title('Autoencoder+GP fit')
        cl_ratio = normFactor*x_decoded[0]/PkOriginal[i]
        relError = 100*np.abs(cl_ratio - 1)

        plt.plot(ls, cl_ratio, alpha=.35, lw = 1.5)

        # plt.xscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$C_l^{GPAE}$/$C_l^{Original}$')
        # plt.legend()
        plt.tight_layout()

        PlotSampleID = 10

        if(i == PlotSampleID):


            plt.figure(99, figsize=(8,6))
            plt.title('Autoencoder+GP fit')
            plt.plot(ls, normFactor * x_test[::].T, 'gray', alpha=0.3)

            plt.plot(ls, normFactor*x_decoded[0], ls='--', alpha= 1.0, lw = 2, label = 'emulated')
            plt.plot(ls, PkOriginal[i], ls='--', alpha=1.0, lw = 2, label = 'original')

            # plt.xscale('log')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$C_l$')
            plt.legend()
            plt.tight_layout()

            ErrTh = 10
            plt.plot(ls[relError > ErrTh], normFactor*x_decoded[0][relError > ErrTh], 'gx', alpha=0.2, label='bad eggs', markersize = '3')
            plt.savefig('../Cl_data/Plots/GP_AE_output.png')


        print(i, 'ERR0R min max (per cent):', np.array([(relError).min(), (relError).max()]) )


    plt.axhline(y=1, ls='-.', lw=1.5)
    plt.savefig('../Cl_data/Plots/GP_AE_ratio.png')

    plt.show()


# ------------------------------------------------------------------------------
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