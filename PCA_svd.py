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

density_file = '../Pk_data/Pk5.npy'
halo_para_file = '../Pk_data/Para5.npy'
pk = pk_load.density_profile(data_path = density_file, para_path = halo_para_file)
k = np.load('../Pk_data/k5.npy')


(x_train, y_train), (x_test, y_test) = pk.load_data()

y = x_train[:1000,:].T
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


plt.figure(900)
# plt.scatter(K[:,0], K[:,1])
plt.scatter(W[0], W[1], c = y_test[:1000][:,4])
plt.savefig('../Pk_data/SVDvsVAE/ScatterW.png')
