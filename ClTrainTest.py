'''
CAMB -> io
'''

import numpy as np
import matplotlib.pylab as plt
import glob


nbins = 2551 # no. of k values.


# data_path = '../Pk_data/CosmicEmu-master/P_cb/EMU*.txt'

data_path = '../Cl_data/totCL*.npy'

Allfiles = sorted(glob.glob(data_path))

Cl = np.zeros(shape= (len(Allfiles), nbins))
l = np.zeros(shape= (len(Allfiles), nbins))

for i in range(len(Allfiles)):
    Cl[i] = np.load(Allfiles[i])[:,0]


PlotSample = True
if PlotSample:
    for i in range(10):
        plt.figure(10)
        plt.plot(Cl[i])
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('l')
        plt.ylabel('C_l')

    plt.show()



AllPara = np.load('../Cl_data/Para5.npy')

plt.figure(43)
plt.scatter(AllPara[:, :5][:, 2], AllPara[:, :5][:, 1], c = AllPara[:, :5][:, 3])
plt.show()

# OmegaM, Omegab, sigma8, h, ns, w0, wb, OmegaNu, z


# np.save('../Cl_data/k5Test32.npy', k[0])
np.save('../Cl_data/Cl.npy', Cl)
# np.save('../Cl_data/Para5Test32.npy', AllPara[:,:5])
