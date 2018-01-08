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


totalFiles = 1000

# ------------------------------------------------------------------------------


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


x_train = x_train/normFactor
x_test = x_test/normFactor
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
assert np.allclose(Pxx, np.dot(U, np.dot(np.diag(s), Vh)))

NoEigenComp = 2

TruncU = U[:, :NoEigenComp]     #Truncation
TruncS = s[:NoEigenComp]
TruncSq = np.diag(TruncS)
TruncVh = Vh[:NoEigenComp,:]

K = np.matmul(TruncU, TruncSq)/np.sqrt(NoEigenComp)
W1 = np.sqrt(NoEigenComp)*np.matmul(np.diag(1./TruncS), TruncU.T)
W = np.matmul(W1, y)

Pred = np.matmul(K,W)

k = np.load('../Pk_data/SVDvsVAE/k5.npy')
for i in range(30, 40):
    plt.figure(10)
    plt.plot(k, y[:,i]*stdy + yRowMean, 'x', label = 'data')
    plt.plot(k, Pred[:,i]*stdy + yRowMean, 'k')

    plt.xscale('log')
    plt.yscale('log')

plt.show()

plt.figure(4343)
plt.plot(s, 'o-')
plt.xlim(-1,)
plt.ylim(-1,45)
plt.savefig('../Pk_data/SVDvsVAE/svd_fig1.png')

np.savetxt('../Pk_data/SVDvsVAE/K_for2Eval.txt', K)   # Basis
np.savetxt('../Pk_data/SVDvsVAE/W_for2Eval.txt', W)   # Weights

np.savetxt('../Pk_data/SVDvsVAE/stdy.txt', [stdy])
np.savetxt('../Pk_data/SVDvsVAE/yRowMean.txt',yRowMean)


plt.figure(900, figsize=(7,6))
plt.title('Truncated PCA weights')
plt.xlabel('W[0]')
plt.ylabel('W[1]')
CS = plt.scatter(W[0], W[1], c = y_train[:,0], s = 15, alpha=0.6)
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel(r'$\Omega_m$')
plt.tight_layout()
plt.savefig('../Pk_data/SVDvsVAE/SVD_TruncatedWeights.png')
