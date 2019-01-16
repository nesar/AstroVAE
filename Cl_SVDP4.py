'''
SVD like K, W -> weights

Save W

K * W -> Prediction

Gotta save RowMean, std.dev etc

'''


import numpy as np
import scipy.linalg as SL
import matplotlib.pyplot as plt
# import Cl_load
# import SetPub
# SetPub.set_pub()
import params

# num_train = 256
# num_test = 16

NoEigenComp = 32


# ----------------------------- i/o ------------------------------------------

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

#------------------------- SCALAR parameter for rescaling -----------------------
#### ---- All the Cl's are rescaled uniformly #####################

# minVal = np.min( [np.min(x_train), np.min(x_test ) ])
# meanFactor = 1.1*minVal if minVal < 0 else 0
# # meanFactor = 0.0
# print('-------mean factor:', meanFactor)
# x_train = x_train - meanFactor #/ 255.
# x_test = x_test - meanFactor #/ 255.
#
# # x_train = np.log10(x_train) #x_train[:,2:] #
# # x_test =  np.log10(x_test) #x_test[:,2:] #
#
# normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
# # normFactor = 1
# print('-------normalization factor:', normFactor)
# x_train = x_train.astype('float32')/normFactor #/ 255.
# x_test = x_test.astype('float32')/normFactor #/ 255.
#
#
# np.savetxt(DataDir+'meanfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt', [meanFactor])
# np.savetxt(DataDir+'normfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt', [normFactor])
#

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
ls = np.loadtxt( DataDir + 'P'+str(num_para)+'ls_'+str(num_train)+'.txt')[2:]



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


plt.figure(231, figsize=(7,6))
plt.title('Truncated PCA weights')
plt.xlabel('W[0]')
plt.ylabel('W[1]')
CS = plt.scatter(K[0], K[1], c = y_train[:,0], s = 15, alpha=0.6)
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel(r'$\Omega_m$')
plt.tight_layout()
plt.savefig(PlotsDir + 'SVD_TruncatedWeightsP4.png')
