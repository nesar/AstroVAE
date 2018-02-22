import numpy as np
import camb
import itertools
from camb import model, initialpower


"""
first 2 outputs from CAMB - totCL and unlensed CL both are 0's. 
CAMBFast maybe better?
CosmoMC works well with CAMB
"""

totalFiles = 1024

para5 = np.loadtxt('../Cl_data/Data/LatinCosmoP4'+str(totalFiles)+'.txt')

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()

#----------- for sigma_8---------------

#Now get matter power spectra and sigma8 at redshift 0 and 0.8
# pars = camb.CAMBparams()
# pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
# pars.set_dark_energy() #re-set defaults
# pars.InitPower.set_params(ns=0.965)
#Not non-linear corrections couples to smaller scales than you want
# pars.set_matter_power(redshifts=[0.], kmax=2.0)
#
#Linear spectra
# pars.NonLinear = model.NonLinear_none
# results = camb.get_results(pars)
# kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
# s8 = np.array(results.get_sigma8())
#
# #Non-Linear spectra (Halofit)
# pars.NonLinear = model.NonLinear_both
# results.calc_power_spectra(pars)
# kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
#
# print(results.get_sigma8())

#---------------------------------------


for i in range(para5.shape[0]):
    print(i)

    pars.set_cosmology(H0=100*para5[i, 2], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0,
                       tau=0.06)
    pars.InitPower.set_params(ns=para5[i, 3], r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);



    #
    # #-------- sigma_8 --------------------------
    # pars.set_matter_power(redshifts=[0.], kmax=2.0)
    # # Linear spectra
    # pars.NonLinear = model.NonLinear_none
    # results = camb.get_results(pars)
    # kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)
    # s8 = np.array(results.get_sigma8())
    #
    # # Non-Linear spectra (Halofit)
    # pars.NonLinear = model.NonLinear_both
    # results.calc_power_spectra(pars)
    # kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1,
    #                                                                    npoints=200)
    #
    # sigma8_camb = results.get_sigma8()  # present value of sigma_8 --- check kman, mikh etc
    # #---------------------------------------------------
    #
    # sigma8_input = para5[i, 2]
    #
    # r = (sigma8_input ** 2) / (sigma8_camb ** 2) # rescale factor
    r = 1
    # #---------------------------------------------------


    #calculate results for these parameters
    results = camb.get_results(pars)

    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    totCL = powers['total']*r
    unlensedCL = powers['unlensed_scalar']*r

    np.save('../Cl_data/Data/LatintotCLP4'+str(totalFiles)+'_'+str(i) +'.npy', totCL)
    np.save('../Cl_data/Data/LatinunlensedCLP4'+str(totalFiles)+'_'+str(i)+'.npy', unlensedCL)

ls = np.arange(totCL.shape[0])

np.save('../Cl_data/Data/LatinPara5P4_'+str(totalFiles)+'.npy', para5)
np.save('../Cl_data/Data/LatinlsP4_'+str(totalFiles)+'.npy', ls)


