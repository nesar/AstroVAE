import numpy as np
import scipy.linalg as SL
import matplotlib.pyplot as plt


import george

def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)

from george.kernels import Matern32Kernel
kernel = Matern32Kernel(0.5, ndim=4)


totalFiles = 256
TestFiles = 16

NoEigenComp = 16


# ----------------------------- i/o ------------------------------------------

DataDir = '../Cl_data/Data/'
PlotsDir = '../Cl_data/Plots/'
ModelDir = '../Cl_data/Model/'





Mod16 = np.load('../Cl_data/Data/LatinCl_16Mod.npy')
Mod256 = np.load('../Cl_data/Data/LatinCl_256Mod.npy')

Para256 = np.loadtxt('../Cl_data/Data/para4_train.txt')
Para16 = np.loadtxt('../Cl_data/Data/para4_new.txt')

normFactor = np.max( [np.max(Mod256), np.max(Mod16) ])
# normFactor = 1
print('-------normalization factor:', normFactor)

x_train = Mod256.astype('float32')/normFactor #/ 255.
x_test = Mod16.astype('float32')/normFactor #/ 255.

x_train = x_train[:,2:]
x_test = x_test[:,2:]

# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


y = x_train.T
yRowMean = np.zeros_like(y[:,0])

for i in range(y.shape[0]):
    yRowMean[i] = np.mean(y[i])

for i in range( y[0].shape[0] ):
    y[:,i] = (y[:,i] - yRowMean)

stdy = np.std(y)
y = y/stdy

Pxx = y
U, s, Vh = SL.svd(Pxx, full_matrices=False)
# assert np.allclose(Pxx, np.dot(U, np.dot(np.diag(s), Vh)))
print np.abs(Pxx - (np.dot(U, np.dot(np.diag(s), Vh))) ).max()



TruncU = U[:, :NoEigenComp]     #Truncation
TruncS = s[:NoEigenComp]
TruncSq = np.diag(TruncS)
TruncVh = Vh[:NoEigenComp,:]

K = np.matmul(TruncU, TruncSq)/np.sqrt(NoEigenComp)
W1 = np.sqrt(NoEigenComp)*np.matmul(np.diag(1./TruncS), TruncU.T)
W = np.matmul(W1, y)

Pred = np.matmul(K,W)

# ls = np.load('../Cl_data/k5.npy')
ls = np.load(DataDir + 'LatinlsP4_' + str(TestFiles) + '.npy')[2:]
for i in range(10, 20):
    plt.figure(10)
    plt.plot(ls, y[:,i]*stdy + yRowMean, 'r--', label = 'data', alpha = 0.7)
    plt.plot(ls, Pred[:,i]*stdy + yRowMean, 'b', alpha = 0.3)

    # plt.xscale('log')
    # plt.yscale('log')

plt.show()

plt.figure(4343)
plt.plot(s, 'o-')
plt.xlim(-1,100)
plt.ylim(-1,45)
plt.savefig(PlotsDir + 'svd_fig1.png')

np.savetxt(DataDir + 'K_for2EvalP4.txt', K)   # Basis
np.savetxt(DataDir + 'W_for2EvalP4.txt', W)   # Weights

np.savetxt(DataDir + 'stdyP4.txt', [stdy])
np.savetxt(DataDir + 'yRowMeanP4.txt',yRowMean)


plt.figure(900, figsize=(7,6))
plt.title('Truncated PCA weights')
plt.xlabel('W[0]')
plt.ylabel('W[1]')
CS = plt.scatter(W[0], W[1], c = Para256[:,0], s = 15, alpha=0.6)
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel(r'$\Omega_m$')
plt.tight_layout()
plt.savefig(PlotsDir + 'SVD_TruncatedWeightsP4.png')








#################### GP FIT ############################
# ------------------------------------------------------

hmf = Mod256


X1 = hmf[:, 0][:, np.newaxis]
X1a = rescale01(np.min(X1), np.max(X1), X1)

X2 = hmf[:, 1][:, np.newaxis]
X2a = rescale01(np.min(X2), np.max(X2), X2)

X3 = hmf[:, 2][:, np.newaxis]
X3a = rescale01(np.min(X3), np.max(X3), X3)

X4 = hmf[:, 3][:, np.newaxis]
X4a = rescale01(np.min(X4), np.max(X4), X4)

# X5 = hmf[:, 4][:, np.newaxis]
# X5a = rescale01(np.min(X5), np.max(X5), X5)

y = np.loadtxt(DataDir + 'W_for2EvalP4.txt', dtype=float)  #

XY = np.array(np.array([X1a, X2a, X3a, X4a])[:, :, 0])[:, np.newaxis]

# ------------------------------------------------------------------------------

RealPara = np.array([0.13, 0.022, 0.75, 1.01])
RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2])
RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
# RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])

test_pts = RealPara.reshape(4, -1).T


# ------------------------------------------------------------------------------

W_pred = np.array([np.zeros(shape=NoEigenComp)])
gp={}
for i in range(NoEigenComp):
    gp["fit{0}".format(i)]= george.GP(kernel)
    gp["fit{0}".format(i)].compute(XY[:, 0, :].T)
    W_pred[:,i] = gp["fit{0}".format(i)].predict(y[i], test_pts)[0]

# ------------------------------------------------------------------------------



