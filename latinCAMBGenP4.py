import numpy as np
import camb
import itertools
from camb import model, initialpower


"""
first 2 outputs from CAMB - totCL and unlensed CL both are 0's. 
CAMBFast maybe better?
CosmoMC works well with CAMB
"""

numpara = 5
ndim = 2551
totalFiles = 16
lmax = 2500

para5 = np.loadtxt('../Cl_data/Data/LatinCosmoP5'+str(totalFiles)+'.txt')

# print(para5)

#
#Set up a new set of parameters for CAMB
#
# Get CMB power spectra, as requested by the spectra argument. All power spectra are l(l+1)C_l/2pi
# self owned numpy arrays (0..lmax, 0..3), where 0..3 index are TT, EE, BB TT,
# unless raw_cl is True in which case return just C_l.
# For the lens_potential the power spectrum returned is that of the deflection.

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
AllTT = np.zeros(shape=(totalFiles, numpara + ndim) ) # TT
AllEE = np.zeros(shape=(totalFiles, numpara + ndim) ) #
AllBB = np.zeros(shape=(totalFiles, numpara + ndim) )
AllTE = np.zeros(shape=(totalFiles, numpara + ndim) )

for i in range(totalFiles):
    print(para5[i])

    pars = camb.CAMBparams()

    pars.set_cosmology(H0=100*para5[i, 2], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0,
                       tau=0.06)
    pars.InitPower.set_params(ns=para5[i, 3], r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0);




    #-------- sigma_8 --------------------------
    pars.set_matter_power(redshifts=[0.], kmax=2.0)
    # Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)
    s8 = np.array(results.get_sigma8())

    # Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1,
                                                                       npoints=200)

    sigma8_camb = results.get_sigma8()  # present value of sigma_8 --- check kman, mikh etc
    #---------------------------------------------------

    sigma8_input = para5[i, 2]

    r = (sigma8_input ** 2) / (sigma8_camb ** 2) # rescale factor
    # r = 1
    # #---------------------------------------------------


    #calculate results for these parameters
    results = camb.get_results(pars)

    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    totCL = powers['total']*r
    unlensedCL = powers['unlensed_scalar']*r

    AllTT[i] = np.hstack([para5[i], totCL[:,0] ])
    AllEE[i] = np.hstack([para5[i], totCL[:,1] ])
    AllBB[i] = np.hstack([para5[i], totCL[:,2] ])
    AllTE[i] = np.hstack([para5[i], totCL[:,3] ])


    # np.save('../Cl_data/Data/LatintotCLP4'+str(totalFiles)+'_'+str(i) +'.npy', totCL)
    # np.save('../Cl_data/Data/LatinunlensedCLP4'+str(totalFiles)+'_'+str(i)+'.npy', unlensedCL)

ls = np.arange(totCL.shape[0])

# np.save('../Cl_data/Data/LatinPara5P4_'+str(totalFiles)+'.npy', para5)
np.savetxt('../Cl_data/Data/P5ls_'+str(totalFiles)+'.txt', ls)

np.savetxt('../Cl_data/Data/P5TTCl_'+str(totalFiles)+'.txt', AllTT)
np.savetxt('../Cl_data/Data/P5EECl_'+str(totalFiles)+'.txt', AllEE)
np.savetxt('../Cl_data/Data/P5BBCl_'+str(totalFiles)+'.txt', AllBB)
np.savetxt('../Cl_data/Data/P5TECl_'+str(totalFiles)+'.txt', AllTE)

