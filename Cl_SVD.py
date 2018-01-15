'''
SVD like K, W -> weights

Save W

K * W -> Prediction

Gotta save RowMean, std.dev etc

'''


import numpy as np
import scipy.linalg as SL
import matplotlib.pyplot as plt
import pk_load
import SetPub
SetPub.set_pub()


nsize = 2
totalFiles = nsize**5 #32

NoEigenComp = 10


# ------------------------------------------------------------------------------


density_file = '../Cl_data/Cl_'+str(nsize)+'.npy'
halo_para_file = '../Cl_data/Para5_'+str(nsize)+'.npy'
pk = pk_load.density_profile(data_path = density_file, para_path = halo_para_file)

(x_train, y_train), (x_test, y_test) = pk.load_data()

x_train = x_train[:totalFiles][:,2:]
x_test = x_test[:np.int(0.2*totalFiles)][:,2:]
y_train = y_train[:totalFiles]
y_test = y_test[:np.int(0.2*totalFiles)]


print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')

normFactor = np.max( [np.max(x_train), np.max(x_test ) ])


x_train = x_train/normFactor
x_test = x_test/normFactor
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

np.save('../Cl_data/normfactor_'+str(nsize)+'.npy', normFactor)

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
assert np.allclose(Pxx, np.dot(U, np.dot(np.diag(s), Vh)))


TruncU = U[:, :NoEigenComp]     #Truncation
TruncS = s[:NoEigenComp]
TruncSq = np.diag(TruncS)
TruncVh = Vh[:NoEigenComp,:]

K = np.matmul(TruncU, TruncSq)/np.sqrt(NoEigenComp)
W1 = np.sqrt(NoEigenComp)*np.matmul(np.diag(1./TruncS), TruncU.T)
W = np.matmul(W1, y)

Pred = np.matmul(K,W)

# ls = np.load('../Cl_data/k5.npy')
ls = np.load('../Cl_data/ls_' + str(nsize) + '.npy')[2:]
for i in range(10, 20):
    plt.figure(10)
    plt.plot(ls, y[:,i]*stdy + yRowMean, 'x', label = 'data')
    plt.plot(ls, Pred[:,i]*stdy + yRowMean, 'k')

    plt.xscale('log')
    plt.yscale('log')

plt.show()

plt.figure(4343)
plt.plot(s, 'o-')
plt.xlim(-1,)
plt.ylim(-1,45)
plt.savefig('../Cl_data/svd_fig1.png')

np.savetxt('../Cl_data/K_for2Eval.txt', K)   # Basis
np.savetxt('../Cl_data/W_for2Eval.txt', W)   # Weights

np.savetxt('../Cl_data/stdy.txt', [stdy])
np.savetxt('../Cl_data/yRowMean.txt',yRowMean)


plt.figure(900, figsize=(7,6))
plt.title('Truncated PCA weights')
plt.xlabel('W[0]')
plt.ylabel('W[1]')
CS = plt.scatter(W[0], W[1], c = y_train[:,0], s = 15, alpha=0.6)
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel(r'$\Omega_m$')
plt.tight_layout()
# plt.savefig('../Pk_data/SVDvsVAE/SVD_TruncatedWeights.png')
