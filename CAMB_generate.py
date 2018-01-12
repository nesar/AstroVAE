import numpy as np
import camb
import itertools



nsize = 2

OmegaM = np.linspace(0.12, 0.155, nsize)
# OmegaM = normalGenerate(0.12, 0.155, nsize)  # Not sure if the data should be generated likewise?
Omegab = np.linspace(0.0215, 0.0235, nsize)
sigma8 = np.linspace(0.7, 0.9, nsize)
# sigma8 = 0.8*np.ones(shape=nsize)
h = np.linspace(0.55, 0.85, nsize)
ns = np.linspace(0.85, 1.05, nsize)

# allGrid = np.array(np.meshgrid(OmegaM, Omegab, sigma8, h, ns))

para5 = np.array(list(itertools.product(OmegaM, Omegab, sigma8, h, ns)))

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()

for i in range(para5.shape[0]):

    pars.set_cosmology(H0=100*para5[i, 3], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(ns=para5[i, 4], r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);

    #calculate results for these parameters
    results = camb.get_results(pars)

    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    totCL = powers['total']
    unlensedCL = powers['unlensed_scalar']

    np.save('../Cl_data/totCL'+str(i) + '.npy', totCL)
    np.save('../Cl_data/unlensedCL'+str(i)+'.npy', unlensedCL)

ls = np.arange(totCL.shape[0])

np.save('../Cl_data/Para5.npy', para5)
np.save('../Cl_data/ls.npy', ls)


