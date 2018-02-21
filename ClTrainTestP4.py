'''
CAMB -> io
'''

import numpy as np
import matplotlib.pylab as plt
import glob

import SetPub
SetPub.set_pub()



nbins = 2551 # no. of k values.
# nsize = 2
totalFiles = 256

# data_path = '../Pk_data/CosmicEmu-master/P_cb/EMU*.txt'
# data_path = '../Cl_data/totCL'+str(nsize)+'*.npy'
data_path = '../Cl_data/Data/LatintotCLP4'+str(totalFiles)+'*.npy'


Allfiles = sorted(glob.glob(data_path))

Cl = np.zeros(shape= (len(Allfiles), nbins))
l = np.zeros(shape= (len(Allfiles), nbins))

for i in range(len(Allfiles)):
    Cl[i] = np.load(Allfiles[i])[:,0]


ls = np.arange(Cl[0].shape[0])
np.save('../Cl_data/Data/ls_'+str(totalFiles)+'.npy', ls)


PlotSample = True
if PlotSample:
    for i in range( (totalFiles)):
        plt.figure(10)
        plt.plot(ls[2:], Cl[i,2:], alpha = 0.3)
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$C_l$')
    plt.tight_layout()

    plt.show()


# AllPara = np.load('../Cl_data/Para5_'+str(totalFiles)+'.npy')
AllPara = np.load('../Cl_data/Data/LatinPara5P4_'+str(totalFiles)+'.npy')

plt.figure(43)
plt.scatter(AllPara[:, :5][:, 2], AllPara[:, :5][:, 1], c = AllPara[:, :5][:, 3])
plt.show()

# OmegaM, Omegab, sigma8, h, ns, w0, wb, OmegaNu, z

# np.save('../Cl_data/Cl_'+str(totalFiles)+'.npy', Cl)#[:,2:])
np.save('../Cl_data/Data/LatinClP4_'+str(totalFiles)+'.npy', Cl)#[:,2:])
