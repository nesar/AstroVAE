import numpy as np
import matplotlib.pylab as plt


nsize = 100
for i in range(nsize):
    fileIn = '../Pk_data/CosmicEmu-master/P_cb/EMU'

    Pk, k = np.loadtxt(fileIn + str(1000*i) + '.txt').T

    plt.figure(10)
    plt.plot(Pk, k, alpha = 0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')

plt.show()

OmegaM, Omegab, sigma8, h, ns, w0, wb, OmegaNu, z = np.loadtxt(
    '../Pk_data/CosmicEmu-master/P_cb/xstar.dat').T


