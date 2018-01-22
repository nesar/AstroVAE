import numpy as np
import camb
import itertools

"""
first 2 outputs from CAMB - totCL and unlensed CL both are 0's. 
CAMBFast maybe better?
CosmoMC works well with CAMB
"""

totalFiles = 256

para5 = np.loadtxt('../Cl_data/Data/LatinCosmo'+str(totalFiles)+'.txt')

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()

#----------- for sigma_8---------------
# results = camb.get_results(pars)
# kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
# s8 = np.array(results.get_sigma8())
# print(results.get_sigma8())

#---------------------------------------


for i in range(para5.shape[0]):
    print(i)

    pars.set_cosmology(H0=100*para5[i, 3], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(ns=para5[i, 4], r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);

    #calculate results for these parameters
    results = camb.get_results(pars)

    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    totCL = powers['total']
    unlensedCL = powers['unlensed_scalar']

    np.save('../Cl_data/Data/LatintotCL'+str(totalFiles)+'_'+str(i) +'.npy', totCL)
    np.save('../Cl_data/Data/LatinunlensedCL'+str(totalFiles)+'_'+str(i)+'.npy', unlensedCL)

ls = np.arange(totCL.shape[0])

np.save('../Cl_data/Data/LatinPara5_'+str(totalFiles)+'.npy', para5)
np.save('../Cl_data/Data/Latinls_'+str(totalFiles)+'.npy', ls)


