'''
SVD like K, W -> weights

Save W

K * W -> Prediction

Gotta save RowMean, std.dev etc

'''


import numpy as np
import scipy.linalg as SL
import matplotlib.pyplot as plt
import Cl_load
# import SetPub
# SetPub.set_pub()


totalFiles = 256
TestFiles = 16

NoEigenComp = 16


# ----------------------------- i/o ------------------------------------------

DataDir = '../Cl_data/Data/'
PlotsDir = '../Cl_data/Plots/'
ModelDir = '../Cl_data/Model/'


train_path = DataDir + 'LatinClP4_'+str(totalFiles)+'.npy'
train_target_path =  DataDir + 'LatinCosmoP4'+str(totalFiles)+'.txt'
test_path = DataDir + 'LatinClP4_'+str(TestFiles)+'.npy'
test_target_path =  DataDir + 'LatinCosmoP4'+str(TestFiles)+'.txt'


# camb_in = Cl_load.cmb_profile(train_path = train_path,  train_target_path = train_target_path , test_path = test_path, test_target_path = test_target_path, num_para=5)


# (x_train, y_train), (x_test, y_test) = camb_in.load_data()


x_train = np.load(train_path)
y_train = np.load(train_target_path)
x_test = np.load(test_path)
y_test = np.load(test_target_path)


# Mod16 = np.load('../Cl_data/Data/LatinCl_16Mod.npy')
# Mod256 = np.load('../Cl_data/Data/LatinCl_256Mod.npy')
#
# Para256 = np.loadtxt('../Cl_data/Data/para4_train.txt')
# Para16 = np.loadtxt('../Cl_data/Data/para4_new.txt')
#
# x_train = Mod256
# x_test = Mod16
# y_train = Para256
# y_test = Para16



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
CS = plt.scatter(W[0], W[1], c = y_train[:,0], s = 15, alpha=0.6)
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel(r'$\Omega_m$')
plt.tight_layout()
plt.savefig(PlotsDir + 'SVD_TruncatedWeightsP4.png')
