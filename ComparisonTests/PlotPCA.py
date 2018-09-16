import matplotlib.pylab as plt
import numpy as np
from numpy import genfromtxt


### PCA ###
pca_basis = genfromtxt('PCA_data/PCA_basis.csv', delimiter=',')
pca_weights = genfromtxt('PCA_data/PCA_test_weights.csv', delimiter=',')

print np.shape(pca_basis)
print np.shape(pca_weights)

pca_pred = np.matmul(pca_weights.T, pca_basis)

print pca_pred

plt.figure(12)
plt.plot(pca_pred.T, 'k')



### VAE ###

vae_pred = np.loadtxt('TTPred25_VAE.txt')
plt.plot(vae_pred.T, 'r')



### CAMB ###

camb_TT = np.loadtxt('TTtrue.txt')
ls = np.loadtxt('ls.txt')


plt.plot(camb_TT.T, 'b')

plt.show()



