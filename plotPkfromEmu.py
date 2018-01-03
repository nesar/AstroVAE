import numpy as np
import matplotlib.pylab as plt
import glob


nbins = 351

data_path = '../Pk_data/CosmicEmu-master/P_cb/EMU*.txt'

Allfiles = sorted(glob.glob(data_path))

Pk = np.zeros(shape= (len(Allfiles), nbins))
k = np.zeros(shape= (len(Allfiles), nbins))

for i in range(len(Allfiles)):

    Pk[i], k[i] = np.loadtxt(Allfiles[i]).T

## Not all k's are same for each Pk (!). Although the difference is quite small - Gonna roll with it.

for i in range(10):
    plt.figure(10)
    plt.plot(Pk[i], k[i])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')

plt.show()

OmegaM, Omegab, sigma8, h, ns, w0, wb, OmegaNu, z = np.loadtxt(
    '../Pk_data/CosmicEmu-master/P_cb/xstar.dat').T


