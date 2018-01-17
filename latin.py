# /home/nes/MEGA/Google_drive/KU courses/Spring2017/SAMSI_May/AllV/latin.py


''' Latin hypercube design
https://pythonhosted.org/pyDOE/randomized.html

'''

import numpy as np
from matplotlib import pyplot as plt
import pyDOE as pyDOE

from scipy.stats.distributions import norm


def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


import SetPub

SetPub.set_pub()

nsize = 2
totalFiles = nsize**5 #32

# OmegaM = np.linspace(0.12, 0.155, totalFiles)
# Omegab = np.linspace(0.0215, 0.0235, totalFiles)
# # sigma8 = np.linspace(0.7, 0.9, totalFiles)
# # sigma8 = 0.8*np.ones(shape=totalFiles)
# sigma8 = np.linspace(0.799, 0.8001, totalFiles)  # Dunno how to set sigma_8 in CAMB yet
# h = np.linspace(0.55, 0.85, totalFiles)
# ns = np.linspace(0.85, 1.05, totalFiles)


OmegaM = np.linspace(0.12, 0.155, totalFiles)
Omegab = np.linspace(0.0215, 0.0235, totalFiles)
sigma8 = np.linspace(0.799, 0.8001, totalFiles)  # Dunno how to set sigma_8 in CAMB yet
h = np.linspace(0.55, 0.85, totalFiles)
ns = np.linspace(0.85, 1.05, totalFiles)

AllPara = np.vstack([OmegaM, Omegab, sigma8, h, ns])

lhd = pyDOE.lhs(5, samples=totalFiles, criterion='cm')
print lhd
print
# lhd = norm(loc=0, scale=1).ppf(lhd)  # this applies to both factors here


##
f, a = plt.subplots(5, 5, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.rcParams.update({'font.size': 8})

for i in range(5):
    for j in range(i+1):
        a[i, j].scatter(lhd[:, i], lhd[:, j], s=20)
        a[i, j].grid(True)

#plt.savefig('LatinSq.png', figsize=(10, 10))
plt.show()
idx = (lhd * totalFiles).astype(int)

AllCombinations = np.zeros((totalFiles, 5))
for i in range(5):
    AllCombinations[:, i] = AllPara[i][idx[:, i]]

np.savetxt('LatinCosmo.txt', AllCombinations)

print AllCombinations

# for i in range(totalFiles):
# idx = (lhd[:,i]*totalFiles).astype(int)
# AllPara[]


# plt.tight_layout()


#  Can we design lhc such that things
design = pyDOE.lhs(5, samples=10)
from scipy.stats.distributions import norm

means = [1, 2, 3, 4]
stdvs = [0.1, 0.5, 1, 0.25]
for i in xrange(4):
    design[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(design[:, i])

print design
#


