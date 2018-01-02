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

'''

nsize = 100

import numpy as np

OmegaM = np.linspace(0.25, 0.35, nsize)
Omegab = np.linspace(0.25, 0.35, nsize)
sigma8 = np.linspace(0.7, 0.85, nsize)
h = np.linspace(66, 75, nsize)
ns = np.linspace(0.95, 1.0, nsize)
w0 = np.linspace(0.95, 1.0, nsize)
wb = np.linspace(0.95, 1.0, nsize) ## Should be changed later due to w_a's distribution
OmegaNu = np.linspace(0.95, 1.0, nsize)
z = np.linspace(1.4, 1.7, nsize)
 

AllPara = np.vstack([OmegaM, H0, ns, sigma8, delta_h])

np.savetxt('xstar.dat', AllPara)

