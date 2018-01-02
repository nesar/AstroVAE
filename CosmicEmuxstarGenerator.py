'''

0.12  | omega_m   | 0.155
0.0215| omega_b   | 0.0235
0.7   | sigma_8   | 0.9
0.55  | h         | 0.85
0.85  | n_s       | 1.05
-1.3  | w_0       | -0.7
0.3   | -(w_0+w_a)^(1/4) | 1.29
0.0   | omega_nu  | 0.01
0.0   | z         | 2.02


1. NOTE: w0, wb are NOT changed right now
2. We are drawing values from uniform distribution. -- may have to change it to Gaussian or something else.

'''


nsize = 100000

import numpy as np

OmegaM = np.linspace(0.12, 0.155, nsize)
Omegab = np.linspace(0.0215, 0.0235, nsize)
sigma8 = np.linspace(0.7, 0.9, nsize)
h = np.linspace(0.55, 0.85, nsize)
ns = np.linspace(0.85, 1.05, nsize)
# w0 = np.linspace(-1.3, -0.7, nsize)
w0 = -1.0*np.ones(shape=nsize)
# wb = np.linspace(0.3, 1.29, nsize) ## Should be changed later due to w_a's distribution
wb = 0.4*np.ones(shape=nsize)
OmegaNu = np.linspace(0.0, 0.01, nsize)
z = np.linspace(0.0, 2.02, nsize)
 

AllPara = np.vstack([OmegaM, Omegab, sigma8, h, ns, w0, wb, OmegaNu, z])

np.savetxt('../Pk_data/CosmicEmu-master/P_cb/xstar.dat', AllPara.T)

print('-------done--------')