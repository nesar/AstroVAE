import pyccl as ccl
import numpy as np

# Create new Cosmology object with a given set of parameters. This keeps track
# of previously-computed cosmological functions
#cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)






'''

This is on the latest version of camb (1.0.1)

-- several features from the old one aren't working: check https://camb.readthedocs.io/en/latest/CAMBdemo.html



https://www.codecogs.com/latex/eqneditor.php

\Omega_m h^2 : [ 0.12, 0.155] \\
\Omega_b h^2 : [ 0.0215, 0.0235] \\
\sigma_8 : [0.7, 0.9] \\
n_s : [0.85, 1.05] \\
h : [0.55, 0.85] \\
\tau : [0.01, 0.8] \\

N_{eff} : [1, 5] \\
\sum m_\nu : [0, 3] \\
r_{0.05}: [0, 2] \\

\omega_0 : [-1.3, -0.7] \\
\omega_a : [-1.73. 1.28]  \\
\omega_\nu : [0.0, 0.01]



'''

#################### LHC ################


''' Latin hypercube design
https://pythonhosted.org/pyDOE/randomized.html

# /home/nes/MEGA/Google_drive/KU courses/Spring2017/SAMSI_May/AllV/latin.py

'''

import numpy as np
import pyDOE as pyDOE

from scipy.stats.distributions import norm


def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


# import SetPub
# SetPub.set_pub()

totalFiles = 2 #1024
num_para = 10

np.random.seed(17)

PlotAll = False
SaveCls = True
###### NEED TO RECHECK THESE VALUES OMEGAM ~ 0.112

OmegaM = np.linspace(0.10, 0.140, totalFiles)
Omegab = np.linspace(0.0205, 0.0235, totalFiles)
sigma8 = np.linspace(0.7, 0.9, totalFiles)
h = np.linspace(0.55, 0.85, totalFiles)
ns = np.linspace(0.85, 1.05, totalFiles)
Omega0 = np.linspace(-1.3, -0.7, totalFiles)
OmegaA = np.linspace(-1.5, 1.0, totalFiles)
tau = np.linspace(0.01, 0.6, totalFiles)
mnu = np.linspace(0, 3, totalFiles)
neff = np.linspace(1.5, 3.5, totalFiles)

# OmegaA = np.linspace(-1.73, 1.28, totalFiles)
# tau = np.linspace(0.01, 0.8, totalFiles)

#################################################
#################################################

AllLabels = [r'$\tilde{\Omega}_m$', r'$\tilde{\Omega}_b$', r'$\tilde{\sigma}_8$', r'$\tilde{h}$',
             r'$\tilde{n}_s$', r'$\tilde{\Omega}_0$', r'$\tilde{\Omega}_a$', r'$\tilde{\tau}$',
             r'$\sum m_\nu$', r'$N_{eff}$']

AllPara = np.vstack([OmegaM, Omegab, sigma8, h, ns, Omega0, OmegaA, tau, mnu, neff])
print(AllPara)

lhd = pyDOE.lhs(num_para, samples=totalFiles, criterion=None) # c cm corr m
print(lhd)

##
if PlotAll:
	import matplotlib.pylab as plt
	f, a = plt.subplots(num_para, num_para, sharex=True, sharey=True)
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
	plt.rcParams.update({'font.size': 8})

	for i in range(num_para):
		for j in range(i+1):
			print(i,j)
			if(i!=j):
		    		a[i, j].scatter(lhd[:, i], lhd[:, j], s=5)
		    		a[i, j].grid(True)
			else:
		    		a[i, i].text(0.4, 0.4, AllLabels[i], size = 'xx-large')
		    		hist, bin_edges = np.histogram(lhd[:,i], density=True, bins=64)
		    		a[i,i].bar(bin_edges[:-1], hist/hist.max(), width=0.2)
		    		plt.xlim(0,1)
		    		plt.ylim(0,1)


	plt.savefig('../Cl_data/Plots/ExtendedPlots/ExtendedLatinSq.png', figsize=(10, 10))
	plt.show()


idx = (lhd * totalFiles).astype(int)

AllCombinations = np.zeros((totalFiles, num_para))
for i in range(num_para):
    AllCombinations[:, i] = AllPara[i][idx[:, i]]

np.savetxt('../Cl_data/Data/ExtendedLatinCosmoP5'+str(totalFiles)+'.txt', AllCombinations)   #### no
# saving files because the its random everytime


AllCombinations = AllPara.T
np.savetxt('../Cl_data/Data/GridCosmoP5'+str(totalFiles)+'.txt', AllCombinations)

print(AllCombinations)


################## CCL #################################################


cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, n_s=0.96, sigma8 = 0.8, Omega_k = 0,
                      Neff=2, m_nu= 0.2, w0= 1, wa= 0)


# Define a simple binned galaxy number density curve as a function of redshift
z_n = np.linspace(0., 1., 200)
n = np.ones(z_n.shape)

# Create objects to represent tracers of the weak lensing signal with this
# number density (with has_intrinsic_alignment=False)
lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))

# Calculate the angular cross-spectrum of the two tracers as a function of ell
ell = np.arange(2, 10)
cls = ccl.angular_cl(cosmo, lens1, lens2, ell)
print(cls)