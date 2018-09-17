import matplotlib.pylab as plt
import numpy as np
from numpy import genfromtxt

import SetPub
SetPub.set_pub()

### PCA ###
pca_basis = genfromtxt('PCA_data/PCA_basis.csv', delimiter=',')
pca_weights = genfromtxt('PCA_data/PCA_test_weights.csv', delimiter=',')

print np.shape(pca_basis)
print np.shape(pca_weights)

pca_pred = np.matmul(pca_weights.T, pca_basis)

print pca_pred

### VAE ###

vae_pred = np.loadtxt('VAE_data/TTPred25_VAE.txt')

### CAMB ###

camb_TT = np.loadtxt('VAE_data/TTtrue.txt')
ls = np.loadtxt('VAE_data/ls.txt')
params = np.loadtxt('VAE_data/params.txt')



plt.figure(12)
plt.plot(ls, pca_pred.T, 'b--', alpha = 0.5)
plt.plot(ls, vae_pred.T, 'r--', alpha = 0.5)
plt.plot(ls, camb_TT.T, 'k--', alpha = 0.5)

plt.show()




plt.figure(13)
plt.plot(ls, pca_pred.T/camb_TT.T, 'b--', alpha = 0.5, label = 'PCA/CAMB')
#plt.plot(vae_pred.T/camb_TT.T, 'r--', alpha = 0.5, label = 'VAE/CAMB')
#plt.plot(camb_TT.T, 'k--', alpha = 0.5)

plt.show()

###########################################################
###########################################################

plt.figure(997, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom = 0.15)  # set the spacing between axes.
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$l(l+1)C_l/2\pi [\mu K^2]$', fontsize = 15)
# ax0.set_title( r'$\text{' +fileOut + '}$')
ax0.text(0.95, 0.95, 'TT' , horizontalalignment='center', verticalalignment='center',
         transform = ax0.transAxes, fontsize = 20)



ax1.axhline(y= 0, ls='dotted')
ax1.axhline(y=0.002, ls='dashed')
ax1.axhline(y=-0.002, ls='dashed')
ax1.set_ylim(-0.0023, 0.0023)
# ax1.set_yscale('log')



ax1.set_xlabel(r'$l$', fontsize = 15)
ax1.set_ylabel(r'$C_l^{Emu}$/$C_l^{CAMB} - 1$', fontsize = 15)


ax0.plot(ls, camb_TT.T[:, 0], 'k--', alpha=0.5, lw = 0.4,  label = 'CAMB')
ax0.plot(ls, pca_pred.T[:, 0]  , 'r--', alpha= 0.3, lw = 0.4, label = 'Emu')


ax0.plot(ls, camb_TT.T, 'k--', alpha=0.5, lw = 0.4)
ax0.plot(ls, pca_pred.T  , 'r--', alpha= 0.3, lw = 0.4)



ax1.plot(ls, pca_pred.T/camb_TT.T - 1, 'k-', lw = 0.5, label = 'emu/camb', alpha = 0.5)


ax0.legend(loc = 'center right')
#ax1.legend()

#plt.show()

plt.savefig('Plots/PCA_CAMB.pdf')


###########################################################
###########################################################

plt.figure(998, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom = 0.15)  # set the spacing between axes.
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$l(l+1)C_l/2\pi [\mu K^2]$', fontsize = 15)
# ax0.set_title( r'$\text{' +fileOut + '}$')
ax0.text(0.95, 0.95, 'TT' , horizontalalignment='center', verticalalignment='center',
         transform = ax0.transAxes, fontsize = 20)



ax1.axhline(y= 0, ls='dotted')
ax1.axhline(y=0.01, ls='dashed')
ax1.axhline(y=-0.01, ls='dashed')
ax1.set_ylim(-0.012, 0.012)
# ax1.set_yscale('log')



ax1.set_xlabel(r'$l$', fontsize = 15)
ax1.set_ylabel(r'$C_l^{Emu}$/$C_l^{CAMB} - 1$', fontsize = 15)


ax0.plot(ls, camb_TT.T[:, 0], 'k--', alpha=0.5, lw = 0.4,  label = 'CAMB')
ax0.plot(ls, vae_pred.T[:, 0]  , 'r--', alpha= 0.3, lw = 0.4, label = 'Emu')


ax0.plot(ls, camb_TT.T, 'k--', alpha=0.5, lw = 0.4)
ax0.plot(ls, vae_pred.T  , 'r--', alpha= 0.3, lw = 0.4)



ax1.plot(ls, vae_pred.T/camb_TT.T - 1, 'k-', lw = 0.5, label = 'emu/camb', alpha = 0.5)


ax0.legend(loc = 'center right')
#ax1.legend()

#plt.show()

plt.savefig('Plots/VAE_CAMB.pdf')






plt.figure(4232)
plt.scatter(pca_weights[0,:], pca_weights[1,:], c = params[:,0])
plt.scatter(pca_weights[0,:], pca_weights[1,:], c = params[:,0])









