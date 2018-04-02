import numpy as np
import matplotlib.pylab as plt
import glob


#allfiles = glob.glob('./*txt')

#allfiles = ['./BICEP.txt', './WMAP.txt', './SPTpol.txt', './PLANCKlegacy.txt']

dirIn = '../Cl_data/RealData/'
allfiles = ['./WMAP.txt', './SPTpol.txt', './PLANCKlegacy.txt']


#lID = np.array([1, 0, 2, 0])
#ClID = np.array([1, 1, 3, 1])
#emaxID = np.array([1, 2, 4, 2])
#eminID = np.array([1, 2, 4, 2])


lID = np.array([0, 2, 0])
ClID = np.array([1, 3, 1])
emaxID = np.array([2, 4, 2])
eminID = np.array([2, 4, 2])


print allfiles




import numpy as np


for fileID in [0, 1, 2]:
    with open( dirIn + allfiles[fileID]) as f:

				    lines = (line for line in f if not line.startswith('#'))
				    allCl = np.loadtxt(lines, skiprows=1)

    l = allCl[:, lID[fileID] ]
    Cl = allCl[:, ClID[fileID]]
    emax = allCl[:, emaxID[fileID]]
    emin = allCl[:, eminID[fileID]]

    print l.shape


				
    plt.figure(10)
    plt.errorbar(l, Cl, yerr=[emax, emin], fmt='x', label = allfiles[fileID][2:-4], alpha = 0.3, ms=1)


#plt.yscale('log')
#plt.xscale('log')
plt.xlim([0,3500])
plt.ylim([0,8000])
plt.legend()
plt.show()

#print allCl.shape
