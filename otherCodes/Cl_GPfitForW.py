"""
GP fit for W matrix - with only 2 eigenvalues

Uses George - package by Dan Foreman McKay - better integration with his MCMC package.
pip install george  - http://dan.iel.fm/george/current/user/quickstart/

"""
print(__doc__)

# Higdon et al 2008, 2012
# Check David's talk for plots of spectrum, and other things.

# See if we want to emulate mass-density instead?



import numpy as np

from matplotlib import pyplot as plt

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
# ExpSineSquared, DotProduct,
# ConstantKernel)

import george
import Cl_load

import SetPub
SetPub.set_pub()

def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


# import SetPub
# SetPub.set_pub()

totalFiles = 512
TestFiles = 32

NoEigenComp = 20



length_scaleParameter = 1.0
length_scaleBoundMin = 0.1
length_scaleBoundMax = 0.3

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

# hmf = np.loadtxt('Data/HMF_5Para.txt')





# ----------------------------- i/o ------------------------------------------

DataDir = '../Cl_data/Data/'
PlotsDir = '../Cl_data/Plots/'
ModelDir = '../Cl_data/Model/'


train_path = DataDir + 'LatinCl_'+str(totalFiles)+'.npy'
train_target_path =  DataDir + 'LatinPara5_'+str(totalFiles)+'.npy'
test_path = DataDir + 'LatinCl_'+str(TestFiles)+'.npy'
test_target_path =  DataDir + 'LatinPara5_'+str(TestFiles)+'.npy'

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

hmf = y_train

# ------------------------------------------------------------------------------


#
# X1 = hmf[:, 0][:, np.newaxis]
# X1a = rescale01(np.min(X1), np.max(X1), X1)
#
# X2 = hmf[:, 1][:, np.newaxis]
# X2a = rescale01(np.min(X2), np.max(X2), X2)
#
# X3 = hmf[:, 2][:, np.newaxis]
# X3a = rescale01(np.min(X3), np.max(X3), X3)
#
# X4 = hmf[:, 3][:, np.newaxis]
# X4a = rescale01(np.min(X4), np.max(X4), X4)
#
# X5 = hmf[:, 4][:, np.newaxis]
# X5a = rescale01(np.min(X5), np.max(X5), X5)
#
#
# XY = np.array(np.array([X1a, X2a, X3a, X4a, X5a])[:, :, 0])[:, np.newaxis]


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

y = np.loadtxt(DataDir + 'W_for2Eval.txt', dtype=float)  #

XY = np.array(np.array([X1a, X2a, X3a, X4a, X5a])[:, :, 0])[:, np.newaxis]

# ------------------------------------------------------------------------------
# Test Sample
# This part will go inside likelihood -- Anirban
RealPara = np.array([0.13, 0.022, 0.8, 0.75, 1.01])
RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2])
RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])

test_pts = RealPara.reshape(5, -1).T

# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------

W_pred = np.array([np.zeros(shape=NoEigenComp)])
gp={}
for i in range(NoEigenComp):
    gp["fit{0}".format(i)]= george.GP(kernel)
    gp["fit{0}".format(i)].compute(XY[:, 0, :].T)
    W_pred[:,i] = gp["fit{0}".format(i)].predict(y[i], test_pts)[0]

# ------------------------------------------------------------------------------


# W_interpol1 = gp1.predict(y[0], test_pts)  # Equal to number of eigenvalues
# W_interpol2 = gp2.predict(y[1], test_pts)
#
# W_pred = np.array([W_interpol1, W_interpol2])

K = np.loadtxt(DataDir + 'K_for2Eval.txt')
Prediction = np.matmul(K, W_pred.T)

# Plots for comparison ---------------------------

normFactor = np.load(DataDir + 'normfactor_'+str(totalFiles)+'.npy')
stdy = np.loadtxt(DataDir + 'stdy.txt')
yRowMean = np.loadtxt(DataDir + 'yRowMean.txt')

# k = np.load('../Cl_data/k5.npy')
ls = np.load(DataDir + 'ls_' + str(TestFiles) + '.npy')[2:]
# EMU0 = np.loadtxt('../Cl_data/EMU0.txt')[:,1] # Generated from CosmicEmu -- original value

PlotSample = True
if PlotSample:
        nsize0 = 32  ## SAMPLE
        # ls = np.load(DataDir + 'ls_' + str(nsize0) + '.npy')[2:]
        EMU0 = np.load(DataDir+ 'LatinCl_' + str(TestFiles) + '.npy')[0, 2:]  # Generated from
        # CosmicEmu --
    # original value
    #     normFactor = np.load(DataDir + 'normfactor_' + str(TestFiles) + '.npy')

    # for i in range(2):
        plt.figure(91, figsize=(8,6))
        plt.title('Truncated PCA+GP fit')
        plt.plot(ls, normFactor*x_test[::].T, 'gray', alpha=0.3)
        plt.plot(ls, normFactor*(Prediction[:, 0] * stdy + yRowMean), 'b--', lw = 2, alpha=1.0,
                 label='decoded')
        plt.plot(ls, EMU0, 'r--', alpha=1.0, lw = 2, label='original')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$C_l$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(PlotsDir + 'GP_PCA_output.png')

plt.show()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

PlotRatio = True
if PlotRatio:

    PkOriginal = np.load(DataDir + 'LatinCl_'+str(TestFiles)+'.npy')[:,2:] # Generated from
    # CosmicEmu --
    # original
    # value
    RealParaArray = np.load(DataDir + 'LatinPara5_'+str(TestFiles)+'.npy')

    # RealPara = np.array([0.13, 0.022, 0.8, 0.75, 1.01])

    for i in range(np.shape(RealParaArray)[0]):

        RealPara = RealParaArray[i]

        RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
        RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
        RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2])
        RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
        RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])

        test_pts = RealPara[:5].reshape(5, -1).T

        # ------------------------------------------------------------------------------

        W_pred = np.array([np.zeros(shape=NoEigenComp)])
        gp = {}
        for i in range(NoEigenComp):
            gp["fit{0}".format(i)] = george.GP(kernel)
            gp["fit{0}".format(i)].compute(XY[:, 0, :].T)
            W_pred[:, i] = gp["fit{0}".format(i)].predict(y[i], test_pts)[0]

        # ------------------------------------------------------------------------------

        # K = np.loadtxt('../Pk_data/SVDvsVAE/K_for2Eval.txt')
        Prediction = np.matmul(K, W_pred.T)


        plt.figure(94, figsize=(8,6))
        plt.title('Truncated PCA+GP fit')
        plt.plot(ls, normFactor*(Prediction[:, 0] * stdy + yRowMean)/PkOriginal[i], alpha=.9,
                 lw = 1.5)
        # plt.xscale('log')
                # plt.yscale('log')
        plt.xlabel('l')
        plt.ylabel(r'$C_l^{GP}$/$C_l^_{Original}$')
                # plt.legend()
                # plt.tight_layout()
    plt.axhline(y=1, ls = '-.', lw = 1.5)
    plt.savefig(PlotsDir + 'GP_PCA_ratio.png')

    plt.show()