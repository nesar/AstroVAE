import numpy as np
import camb
import itertools
from camb import model, initialpower
import matplotlib.pylab as plt

import time
time0 = time.time()

"""
first 2 outputs from CAMB - totCL and unlensed CL both are 0's. 
CAMBFast maybe better?
CosmoMC works well with CAMB
http://camb.readthedocs.io/en/latest/CAMBdemo.html
https://wiki.cosmos.esa.int/planckpla2015/index.php/CMB_spectrum_%26_Likelihood_Code
"""

numpara = 5
# ndim = 2551
totalFiles =  8
# lmax0 = 2500


lmax0 = 12000   ## something off above 8250
# model.lmax_lensed.value = 8250 by default
ell_max = 10000

# ndim = lmax0 + 1


para5 = np.loadtxt('../Cl_data/Data/LatinCosmoP5'+str(totalFiles)+'.txt')

# print(para5)

f, a = plt.subplots(5, 5, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.rcParams.update({'font.size': 8})


AllLabels = [r'$\tilde{\Omega}_m$', r'$\tilde{\Omega}_b$', r'$\tilde{\sigma}_8$', r'$\tilde{h}$',
             r'$\tilde{n}_s$'] ### n_eff, mass nutrino -- tau

for i in range(5):
    for j in range(i+1):
        print(i,j)
        # a[i,j].set_xlabel(AllLabels[i])
        # a[i,j].set_ylabel(AllLabels[j])
        if(i!=j):
            a[i, j].scatter(para5[:, i], para5[:, j], s=10)
            a[i, j].grid(True)
        else:
            # a[i,i].set_title(AllLabels[i])
            a[i, i].text(0.4, 0.4, AllLabels[i], size = 'xx-large')
            hist, bin_edges = np.histogram(para5[:,i], density=True, bins=64)
            # a[i,i].bar(hist)
            a[i,i].bar(bin_edges[:-1], hist/hist.max(), width=0.2)
            # plt.xlim(0,1)
            # plt.ylim(0,1)

            # n, bins, patches = a[i,i].hist(lhd[:,i], bins = 'auto', facecolor='b', alpha=0.25)
            # a[i, i].plot(lhd[:, i], 'go')

#plt.savefig('LatinSq.png', figsize=(10, 10))


plt.show()

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


# AllLabels = [r'$\tilde{\Omega}_m$', r'$\tilde{\Omega}_b$', r'$\tilde{\sigma}_8$', r'$\tilde{
# h}$', r'$\tilde{n}_s$']


#---------------------------------------
AllTT = np.zeros(shape=(totalFiles, numpara + ell_max + 1) ) # TT
AllEE = np.zeros(shape=(totalFiles, numpara + ell_max + 1) ) #
AllBB = np.zeros(shape=(totalFiles, numpara + ell_max + 1) )
AllTE = np.zeros(shape=(totalFiles, numpara + ell_max + 1) ) # Check if this is actually TE --
# negative
# values and CAMB documentation incorrect.

for i in range(totalFiles):
    print(i, para5[i])

    # Set up a new set of parameters for CAMB
    # pars = camb.CAMBparams()
    camb.set_halofit_version('takahashi')
    # This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    # pars.set_cosmology(H0=67.05, ombh2=0.02225, omch2=0.1198,
    #                    tau=0.079, num_massive_neutrinos=0, mnu=0.0,
    #                    standard_neutrino_neff=3.046)
    # pars.InitPower.set_params(As=2.2065e-9, ns=0.9645)
    # pars.set_for_lmax(ell_max, max_eta_k=12000, lens_potential_accuracy=4);
    # pars.set_accuracy(AccuracyBoost=3, lAccuracyBoost=3, lSampleBoost=3, DoLateRadTruncation=False)
    # pars.AccuratePolarization = True
    # pars.AccurateReionization = True
    # pars.YHe = 0.24
    # # pars.omegan = 0.0006445
    # pars.omegak = 0.
    # pars.set_nonlinear_lensing(True)





    pars = camb.CAMBparams()

    # pars.set_cosmology(H0=100*para5[i, 2], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0,
    #                    tau=0.06)

    pars.set_cosmology(H0=100*para5[i, 3], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0,
                       tau=0.06)


    pars.InitPower.set_params(ns=para5[i, 4], r=0)


    # pars.set_for_lmax(lmax= lmax0, max_eta_k=None, k_eta_fac=12.5 , lens_potential_accuracy=1,
    #                   lens_k_eta_reference = 20000)

    pars.set_for_lmax(lmax = lmax0, max_eta_k=20000, lens_potential_accuracy=4);


    ## THIS IS ONLY FOR ACCURACY,
    ## actual lmax is set in results.get_cmb_power_spectra

    # model.lmax_lensed = 10000  ## doesn't work


    pars.set_accuracy(AccuracyBoost=3, lAccuracyBoost=3, lSampleBoost=3, DoLateRadTruncation=False)

    pars.AccuratePolarization = True
    pars.AccurateReionization = True
    pars.YHe = 0.24
    # pars.omegan = 0.0006445
    pars.omegak = 0.
    pars.set_nonlinear_lensing(True)



    print model.lmax_lensed


    #-------- sigma_8 --------------------------
    pars.set_matter_power(redshifts=[0.], kmax=2.0)
    # Linear spectra
    # pars.NonLinear = model.NonLinear_none
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
    # pars.set_for_lmax(lmax= lmax0, k_eta_fac=2.5 , lens_potential_accuracy=0)  ## THIS IS ONLY


    #calculate results for these parameters
    # results0 = camb.get_results(pars)    ### Why this again??????????


    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=ell_max)
    # powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    # powers0 =results0.get_cmb_power_spectra(pars, CMB_unit='muK')


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
np.savetxt('../Cl_data/Data/P5_1ls_'+str(totalFiles)+'.txt', ls)

np.savetxt('../Cl_data/Data/P5_2TTCl_'+str(totalFiles)+'.txt', AllTT)
np.savetxt('../Cl_data/Data/P5_2EECl_'+str(totalFiles)+'.txt', AllEE)
np.savetxt('../Cl_data/Data/P5_2BBCl_'+str(totalFiles)+'.txt', AllBB)
np.savetxt('../Cl_data/Data/P5_2TECl_'+str(totalFiles)+'.txt', AllTE)

time1 = time.time()
print('camb time:', time1 - time0)


plt.figure(32)
plt.plot(AllEE[:, 7:].T)
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$C_l$')
plt.xlabel('$l$')
plt.show()