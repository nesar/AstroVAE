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

    k[i], Pk[i] = np.loadtxt(Allfiles[i]).T


PlotSample = True
if PlotSample:
    for i in range(10):
        plt.figure(10)
        plt.plot(k[i], Pk[i])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('k')
        plt.ylabel('P(k)')

    plt.show()



AllPara = np.loadtxt(
    '../Pk_data/CosmicEmu-master/P_cb/xstar_32.dat')

plt.figure(43)
plt.scatter(AllPara[:, :5][:, 2], AllPara[:, :5][:, 1], c = AllPara[:, :5][:, 3])
plt.show()

# OmegaM, Omegab, sigma8, h, ns, w0, wb, OmegaNu, z


np.save('../Pk_data/k5Test32.npy', k[0])
np.save('../Pk_data/Pk5Test32.npy', Pk)
np.save('../Pk_data/Para5Test32.npy', AllPara[:,:5])
