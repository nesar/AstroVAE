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



totalFiles = 32
latent_dim = 2


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
# ------------------------------------------------------------------------------

import pk_load

density_file = '../Cl_data/Cl.npy'
halo_para_file = '../Cl_data/Para5.npy'
pk = pk_load.density_profile(data_path = density_file, para_path = halo_para_file)

(x_train, y_train), (x_test, y_test) = pk.load_data()

x_train = x_train[:totalFiles]
x_test = x_test[:np.int(0.2*totalFiles)]
y_train = y_train[:totalFiles]
y_test = y_test[:np.int(0.2*totalFiles)]


print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')

normFactor = np.max( [np.max(x_train), np.max(x_test ) ])


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
RealPara = np.array([0.13, 0.022, 0.8, 0.75, 1.01])
RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
# RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2]) ## ns = 0.8 constant while training
RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])

test_pts = RealPara.reshape(5, -1).T

# ------------------------------------------------------------------------------
y = np.load('../Cl_data/encoded_xtrain.npy').T

W_pred = np.array([np.zeros(shape=latent_dim)])
gp={}
for i in range(latent_dim):
    gp["fit{0}".format(i)]= george.GP(kernel)
    gp["fit{0}".format(i)].compute(XY[:, 0, :].T)
    W_pred[:,i] = gp["fit{0}".format(i)].predict(y[i], test_pts)[0]

# ------------------------------------------------------------------------------

#  2 gp because of 2 truncated pca basis
# 2 also because 2 latent variables??

# Specify Gaussian Process
# gp1 = GaussianProcessRegressor(kernel=kernels[0])
# gp2 = GaussianProcessRegressor(kernel=kernels[0])



# gp1 = george.GP(kernel)
# gp2 = george.GP(kernel)

# gp1.fit(  XY[:,0,:].T   ,  y[0])
# gp2.fit( XY[:,0,:].T ,  y[1])

# gp1.compute(XY[:, 0, :].T)
# gp2.compute(XY[:, 0, :].T)

# gp2.optimize( XY[:,0,:].T, y[1], verbose=True)

# W_interpol1 = gp1.predict(y[0], test_pts)  # Equal to number of eigenvalues
# W_interpol2 = gp2.predict(y[1], test_pts)

# Why is the dimension of W_interpol1 = (2,1). I need just (1)

# W_pred = np.array([W_interpol1, W_interpol2])
# print(W_pred)

# W_decoder_input = W_pred[:,0,:].T


# ------------------------------------------------------------------------------
# Decoder acts here
# Have to save and load model architecture, and weights

from keras.models import load_model

fileOut = 'Model'
# vae = load_model('../Pk_data/fullAE_' + fileOut + '.hdf5')
encoder = load_model('../Cl_data/Encoder_' + fileOut + '.hdf5')
decoder = load_model('../Cl_data/Decoder_' + fileOut + '.hdf5')
history = np.load('../Cl_data/TrainingHistory_'+fileOut+'.npy')

# generator = Model(decoder_input, _x_decoded_mean)
# x_decoded = generator.predict(W_pred.T[0,:,:])
x_decoded = decoder.predict(W_pred)


ls = np.load('../Cl_data/ls.npy')#[2:]
EMU0 = np.load('../Cl_data/totCL0.npy')[:,0] # Generated from CosmicEmu -- original value
normFactor = np.load('../Cl_data/normfactor.npy')


PlotSample = True
if PlotSample:
    # for i in range(2):
        plt.figure(91, figsize=(8,6))
        plt.title('Autoencoder+GP fit')
        plt.plot(ls, normFactor*x_test[::].T, 'gray', alpha=0.3)
        plt.plot(ls, normFactor*x_decoded[0], 'b--', lw = 2, alpha=1.0, label='decoded')
        plt.plot(ls, EMU0, 'r--', alpha=1.0, lw = 2, label='original')
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('$l$')
        plt.ylabel(r'$C_l$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('../Cl_data/GP_AE_output.png')

plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    epochs =  history[0,:]
    train_loss = history[1,:]
    val_loss = history[2,:]


    fig, ax = plt.subplots(1,1, sharex= True, figsize = (8,6))
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epochs,train_loss, 'o-')
    ax.plot(epochs,val_loss, 'o-')
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train loss','val loss'])
    plt.tight_layout()
    plt.savefig('../Cl_data/Training_loss.png')

plt.show()



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

PlotRatio = True
if PlotRatio:

    PkOriginal = np.load('../Cl_data/ClTest32.npy')[:,:] # Generated from CosmicEmu -- original
    # value
    RealParaArray = np.load('../Cl_data/Para5.npy')

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
        for i in range(latent_dim):
            gp["fit{0}".format(i)] = george.GP(kernel)
            gp["fit{0}".format(i)].compute(XY[:, 0, :].T)
            W_pred[:, i] = gp["fit{0}".format(i)].predict(y[i], test_pts)[0]

        # ------------------------------------------------------------------------------

        x_decoded = decoder.predict(W_pred)


        plt.figure(94, figsize=(8,6))
        plt.title('Autoencoder+GP fit')

        plt.plot(ls, normFactor*x_decoded[0]/PkOriginal[i], alpha=.9, lw = 1.5)

        plt.xscale('log')
        plt.xlabel('k')
        plt.ylabel(r'$P_{GPAE}(k)$/$P_{Original}(k)$')
        # plt.legend()
        plt.tight_layout()
    plt.savefig('../Cl_data/GP_AE_ratio.png')
    plt.axhline(y=1, ls='-.', lw=1.5)

    plt.show()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Cosmic Variance emulator
