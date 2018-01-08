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

def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)



totalFiles = 5000
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

#  2 gp because of 2 truncated pca basis
# 2 also because 2 latent variables??

# Specify Gaussian Process
# gp1 = GaussianProcessRegressor(kernel=kernels[0])
# gp2 = GaussianProcessRegressor(kernel=kernels[0])

gp1 = george.GP(kernel)
gp2 = george.GP(kernel)

# gp1.fit(  XY[:,0,:].T   ,  y[0])
# gp2.fit( XY[:,0,:].T ,  y[1])

gp1.compute(XY[:, 0, :].T)
gp2.compute(XY[:, 0, :].T)

# gp2.optimize( XY[:,0,:].T, y[1], verbose=True)



# ------------------------------------------------------------------------------

# This part will go inside likelihood -- Anirban
RealPara = np.array([0.13, 0.022, 0.8, 0.75, 1.01])
RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2])
RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])

test_pts = RealPara.reshape(5, -1).T



# ------------------------------------------------------------------------------
# Dimension of W_pred is very small


y = np.load('../Pk_data/SVDvsVAE/encoded_xtrain.npy').T
# y = np.loadtxt('Data/W_for2Eval.txt', dtype=float)  #

W_interpol1 = gp1.predict(y[0], test_pts)  # Equal to number of eigenvalues
W_interpol2 = gp2.predict(y[1], test_pts)

# Why is the dimension of W_interpol1 = (2,1). I need just (1)

W_pred = np.array([W_interpol1, W_interpol2])
print(W_pred)

# Just need one decoder input representation, not 2 -- W_pred right now gives 2
W_decoder_input = W_pred[:,0,:].T

# ------------------------------------------------------------------------------
# Decoder acts here
# Have to save and load model architecture, and weights

from keras.models import load_model

fileOut = 'Model'
# vae = load_model('../Pk_data/fullAE_' + fileOut + '.hdf5')
encoder = load_model('../Pk_data/Encoder_' + fileOut + '.hdf5')
decoder = load_model('../Pk_data/Decoder_' + fileOut + '.hdf5')
history = np.load('../Pk_data/TrainingHistory_'+fileOut+'.npy')

# generator = Model(decoder_input, _x_decoded_mean)
# x_decoded = generator.predict(W_pred.T[0,:,:])
x_decoded = decoder.predict(W_decoder_input)


PlotSample = True
k = np.load('../Pk_data/k5.npy')
EMU0 = np.loadtxt('../Pk_data/EMU0.txt')[:,1] # Generated from CosmicEmu -- original value
normFactor = np.load('../Pk_data/SVDvsVAE/normfactor.npy')

if PlotSample:
    # for i in range(2):
        plt.figure(91, figsize=(8,6))
        plt.plot(k, normFactor*x_test[::20].T, 'gray', alpha=0.2)
        plt.plot(k, normFactor * x_decoded[0], 'b--', lw = 2, alpha=1.0, label='decoded')
        plt.plot(k, EMU0, 'r--', alpha=1.0, lw = 2, label='original')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('k')
        plt.ylabel('P(k)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('../Pk_data/SVDvsVAE/GPAE_output.png')

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
    plt.savefig('../Pk_data/SVDvsVAE/Training_loss.png')

plt.show()



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

#
## # Load decoder architecture here
## K = np.loadtxt('Data/K_for2Eval.txt')
#
## # Get W_pred -> decode to P(k)
## Prediction = np.matmul(K, W_pred[:, :, 0])
#

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#
# # Plots for comparison ---------------------------
#
# xlim1 = 10
# xlim2 = 15
# Mass = np.logspace(np.log10(10 ** xlim1), np.log10(10 ** xlim2), 500)[
#        ::10]  # !!!Check if data points [::10] are properly uniform
#
# hmf = np.loadtxt('Data/HMFTestData.txt')[5:]
#
# hmfPara = np.loadtxt('Data/HMF_5Para.txt')
# for i in range(hmfPara[:, 5:].shape[0]):
#     yA = hmfPara[i, 5:].T  # n(M)    -> all values
#     plt.figure(1)
#     plt.plot(Mass, yA, lw=1.5, color="#4682b4", alpha=0.15)
#
#     plt.figure(2)
#     plt.plot(Mass, yA / hmf, lw=1.5, color="#4682b4", alpha=0.3)
#
# stdy = np.loadtxt('Data/stdy.txt')
# yRowMean = np.loadtxt('Data/yRowMean.txt')
#
# plt.figure(1)
# plt.plot(Mass, Prediction[:, 0] * stdy + yRowMean, 'r-', label='data', lw=1.5)
# plt.plot(Mass[::5], hmf[::5], 'kx', lw=100)
#
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('Plots/GP_fit_fig2.png')
#
# plt.figure(2)
# plt.plot(Mass, (Prediction[:, 0] * stdy + yRowMean) / hmf, 'r-', lw=1.5)
# plt.xscale('log')
# # plt.yscale('log')
#
# plt.savefig('Plots/GP_fit_fig1.png')
# plt.show()
