''' Latin hypercube design
https://pythonhosted.org/pyDOE/randomized.html

# /home/nes/MEGA/Google_drive/KU courses/Spring2017/SAMSI_May/AllV/latin.py

'''

import numpy as np
from matplotlib import pyplot as plt
import pyDOE as pyDOE

from scipy.stats.distributions import norm


def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


import SetPub

SetPub.set_pub()

# nsize = 2
# totalFiles = nsize**5 #32
totalFiles = 256

# OmegaM = np.linspace(0.12, 0.155, totalFiles)
# Omegab = np.linspace(0.0215, 0.0235, totalFiles)
# # sigma8 = np.linspace(0.7, 0.9, totalFiles)
# # sigma8 = 0.8*np.ones(shape=totalFiles)
# sigma8 = np.linspace(0.799, 0.8001, totalFiles)  # Dunno how to set sigma_8 in CAMB yet
# h = np.linspace(0.55, 0.85, totalFiles)
# ns = np.linspace(0.85, 1.05, totalFiles)


OmegaM = np.linspace(0.12, 0.155, totalFiles)
Omegab = np.linspace(0.0215, 0.0235, totalFiles)
# sigma8 = np.linspace(0.799, 0.8001, totalFiles)  # Dunno how to set sigma_8 in CAMB yet
# sigma8 = np.linspace(0.7, 0.9, totalFiles)
h = np.linspace(0.55, 0.85, totalFiles)
ns = np.linspace(0.85, 1.05, totalFiles)

AllLabels = [r'$\tilde{\Omega}_m$', r'$\tilde{\Omega}_b$', r'$\tilde{h}$',
             r'$\tilde{n}_s$']

AllPara = np.vstack([OmegaM, Omegab, h, ns])

lhd = pyDOE.lhs(4, samples=totalFiles, criterion='cm')
print(lhd)
print
# lhd = norm(loc=0, scale=1).ppf(lhd)  # this applies to both factors here


##
f, a = plt.subplots(4, 4, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.rcParams.update({'font.size': 8})

for i in range(4):
    for j in range(i+1):
        print(i,j)
        # a[i,j].set_xlabel(AllLabels[i])
        # a[i,j].set_ylabel(AllLabels[j])
        if(i!=j):
            a[i, j].scatter(lhd[:, i], lhd[:, j], s=20)
            a[i, j].grid(True)
        else:
            # a[i,i].set_title(AllLabels[i])
            a[i, i].text(0.4, 0.4, AllLabels[i], size = 'xx-large')
            hist, bin_edges = np.histogram(lhd[:,i], density=True, bins=64)
            # a[i,i].bar(hist)
            a[i,i].bar(bin_edges[:-1], hist/hist.max(), width=0.2)
            plt.xlim(0,1)
            plt.ylim(0,1)

            # n, bins, patches = a[i,i].hist(lhd[:,i], bins = 'auto', facecolor='b', alpha=0.25)
            # a[i, i].plot(lhd[:, i], 'go')

#plt.savefig('LatinSq.png', figsize=(10, 10))
plt.show()
idx = (lhd * totalFiles).astype(int)

AllCombinations = np.zeros((totalFiles, 4))
for i in range(4):
    AllCombinations[:, i] = AllPara[i][idx[:, i]]

np.savetxt('../Cl_data/Data/LatinCosmoP4'+str(totalFiles)+'.txt', AllCombinations)

print(AllCombinations)

# for i in range(totalFiles):
# idx = (lhd[:,i]*totalFiles).astype(int)
# AllPara[]


# plt.tight_layout()


#  Can we design lhc such that mean and std can be provided

