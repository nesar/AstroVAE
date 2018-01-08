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
import numpy as np

def normalGenerate(lim1, lim2, nsize):
    mu = 0.5*(lim2 - lim1)
    sigma = (lim2 - mu)/16
    N01 = np.random.randn(nsize)
    pdf = sigma*N01 + mu   ## N(mu, sigma)
    return pdf
    # return (N02 - lim N01.max() + N01.min())*(lim2 - lim1)/(N01.max() - N01.min())

nsize = 3  # 100000 without combination - iteration



#-----------------------------------------------------------

OmegaM = np.linspace(0.12, 0.155, nsize)
# OmegaM = normalGenerate(0.12, 0.155, nsize)  # Not sure if the data should be generated likewise?
Omegab = np.linspace(0.0215, 0.0235, nsize)
sigma8 = np.linspace(0.7, 0.9, nsize)
h = np.linspace(0.55, 0.85, nsize)
ns = np.linspace(0.85, 1.05, nsize)



###    Parameters below aren't varied for sampling.

# w0 = np.linspace(-1.3, -0.7, nsize)
w0 = -1.0*np.ones(shape=nsize)
# wb = np.linspace(0.3, 1.29, nsize) ## Should be changed later due to w_a's distribution
wb = 0.4*np.ones(shape=nsize)
OmegaNu = np.zeros(shape=nsize)
# OmegaNu = np.linspace(0.0, 0.01, nsize)  # Keeping one value for now
z = np.zeros(shape=nsize)
# z = np.linspace(0.0, 2.02, nsize)  # Keeping one value for z, for now


#-----------------------------------------------------------


AllPara0 = np.vstack([OmegaM, Omegab, sigma8, h, ns, w0, wb, OmegaNu, z])
## AllPara is a simple stack -- no common combinations of parameters. No mix-n-match
# np.savetxt('../Pk_data/CosmicEmu-master/P_cb/xstar.dat', AllPara.T)

#-----------------------------------------------------------


import itertools

# allGrid = np.array(np.meshgrid(OmegaM, Omegab, sigma8, h, ns))

para5 = np.array(list(itertools.product(OmegaM, Omegab, sigma8, h, ns)))


# w0 = np.linspace(-1.3, -0.7, nsize)
w0 = -1.0*np.ones(shape=nsize**5)
# wb = np.linspace(0.3, 1.29, nsize) ## Should be changed later due to w_a's distribution
wb = 0.4*np.ones(shape=nsize**5)
OmegaNu = np.zeros(shape=nsize**5)
z = np.zeros(shape=nsize**5)


restpara = np.vstack([w0, wb, OmegaNu, z])

# AllPara = np.array([0.13, 0.022, 0.8, 0.75, 1.01, -1.0, 0.4, 0.0, 0.0]).T  ## For EMU0 -- just
# one test case

AllPara = np.hstack([para5, restpara.T])

np.savetxt('../Pk_data/CosmicEmu-master/P_cb/xstar.dat', AllPara)


print('-------done--------')
