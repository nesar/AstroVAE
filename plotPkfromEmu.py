'''
## Not all k's are same for each Pk (!). Although the difference is quite small - Gonna roll with it.

'''

import numpy as np
import matplotlib.pylab as plt
import glob


nbins = 351 # no. of k values.


data_path = '../Pk_data/CosmicEmu-master/P_cb/EMU*.txt'

Allfiles = sorted(glob.glob(data_path))

Pk = np.zeros(shape= (len(Allfiles), nbins))
k = np.zeros(shape= (len(Allfiles), nbins))

for i in range(len(Allfiles)):

    Pk[i], k[i] = np.loadtxt(Allfiles[i]).T


PlotSample = False
if PlotSample:
    for i in range(10):
        plt.figure(10)
        plt.plot(Pk[10000*i], k[10000*i])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('k')
        plt.ylabel('P(k)')

    plt.show()

AllPara = np.loadtxt(
    '../Pk_data/CosmicEmu-master/P_cb/xstar.dat')

# OmegaM, Omegab, sigma8, h, ns, w0, wb, OmegaNu, z


np.save('../Pk_data/Pk.npy', Pk)
np.save('../Pk_data/Para9.npy', AllPara)


plt.plot(Pk[100])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('P(k)')