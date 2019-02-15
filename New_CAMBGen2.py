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
from matplotlib import pyplot as plt
import pyDOE as pyDOE

from scipy.stats.distributions import norm


def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


# import SetPub
# SetPub.set_pub()

# nsize = 2
# totalFiles = nsize**5 #32
totalFiles = 4
num_para = 10

np.random.seed(7)

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

############################## CAMB ###############################

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

lmax0 = 12000   ## something off above 8250
# model.lmax_lensed.value = 8250 by default
ell_max = 10000

#lmax0 = 3000   ## something off above 8250 -- sorted now
# model.lmax_lensed.value = 8250 by default
#ell_max = 2500



para5 = np.loadtxt('../Cl_data/Data/ExtendedLatinCosmoP5'+str(totalFiles)+'.txt')


f, a = plt.subplots(num_para, num_para, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.rcParams.update({'font.size': 8})


for i in range(num_para):
    for j in range(i+1):
        print(i,j)
        if(i!=j):
            a[i, j].scatter(para5[:, i], para5[:, j], s=10)
            a[i, j].grid(True)
        else:
            a[i, i].text(0.4, 0.4, AllLabels[i], size = 'xx-large')
            hist, bin_edges = np.histogram(para5[:,i], density=True, bins=64)
            a[i,i].bar(bin_edges[:-1], hist/hist.max(), width=0.2)

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
AllTT = np.zeros(shape=(totalFiles, num_para + ell_max + 1) ) # TT
AllEE = np.zeros(shape=(totalFiles, num_para + ell_max + 1) ) #
AllBB = np.zeros(shape=(totalFiles, num_para + ell_max + 1) )
AllTE = np.zeros(shape=(totalFiles, num_para + ell_max + 1) ) # Check if this is actually TE --
# negative
# values and CAMB documentation incorrect.

for i in range(totalFiles):
    print(i, para5[i])

    # Set up a new set of parameters for CAMB
    # pars = camb.CAMBparams()
    # camb.set_halofit_version('takahashi')     ########## 1.0.1 ISSUE
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

    # pars.set_cosmology(H0=100*para5[i, 3], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0,
    #                    tau=0.06)

    ###### Dynamical DE ############
    # pars.set_cosmology(H0=100*para5[i, 3], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0,
    #                    tau=para5[i, 7])



    ####### Adding neutrinos #########
    pars.set_cosmology(H0=100*para5[i, 3], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=para5[i, 8],
                       omk=0, tau=para5[i, 7], standard_neutrino_neff=para5[i, 9])

    ## add nnu (N_eff, num_massive_neutrinos. Omega_nu is approximated by CAMB
    ## https://camb.readthedocs.io/en/latest/model.html#camb.model.CAMBparams.set_cosmology

    ##### "mnu --sum of neutrino masses (in eV, Omega_nu is calculated approximately from this
    ### assuming neutrinos non-relativistic today). Set the field values directly if you need
    ### finer control or more complex models." ######

    pars.InitPower.set_params(ns=para5[i, 4], r=0)


    ######### DARK ENERGY #############


    # The dark energy model can be changed as in the previous example, or by assigning to pars.DarkEnergy.
    # e.g. use the PPF model
    from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid

    # pars.DarkEnergy = DarkEnergyPPF(w=-1.2, wa=0.2)
    pars.DarkEnergy = DarkEnergyPPF(w=para5[i, 5], wa=para5[i, 6])
    print('w, wa model parameters:\n\n', pars.DarkEnergy)
    # results = camb.get_background(pars)



    # or can also use a w(a) numerical function
    # (note this will slow things down; make your own dark energy class in fortran for best performance)
    # a = np.logspace(-5, 0, 1000)
    # w = -1.2 + 0.2 * (1 - a)
    # pars.DarkEnergy = DarkEnergyPPF()
    # pars.DarkEnergy.set_w_a_table(a, w)
    # print('Table-interpolated parameters (w and wa are set to estimated values at 0):\n\n'
    #       , pars.DarkEnergy)
    # results2 = camb.get_background(pars)
    #
    # rho, _ = results.get_dark_energy_rho_w(a)
    # rho2, _ = results2.get_dark_energy_rho_w(a)
    # plt.plot(a, rho, color='k')
    # plt.plot(a, rho2, color='r', ls='--')
    # plt.ylabel(r'$\rho/\rho_0$')
    # plt.xlabel('$a$')
    # plt.xlim(0, 1)
    # plt.title('Dark enery density');

    ###################################


    # pars.set_for_lmax(lmax= lmax0, max_eta_k=None, k_eta_fac=12.5 , lens_potential_accuracy=1,
    #                   lens_k_eta_reference = 20000)

    pars.set_for_lmax(lmax = lmax0, max_eta_k=20000, lens_potential_accuracy=4);


    ## THIS IS ONLY FOR ACCURACY,
    ## actual lmax is set in results.get_cmb_power_spectra

    # model.lmax_lensed = 10000  ## doesn't work


    pars.set_accuracy(AccuracyBoost=3, lAccuracyBoost=3, lSampleBoost=3, DoLateRadTruncation=False)

    pars.AccuratePolarization = True
    pars.AccurateReionization = True
    pars.YHe = 0.24 ##helium_fraction
    # pars.omegan = 0.0006445
    pars.omegak = 0.
    pars.set_nonlinear_lensing(True)



    # print model.lmax_lensed   ########## 1.0.1 ISSUE


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

    sigma_ratio = (sigma8_input ** 2) / (sigma8_camb ** 2) # rescale factor
    # sigma_ratio = 1
    # #---------------------------------------------------
    # pars.set_for_lmax(lmax= lmax0, k_eta_fac=2.5 , lens_potential_accuracy=0)  ## THIS IS ONLY


    #calculate results for these parameters
    # results0 = camb.get_results(pars)    ### Why this again??????????


    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=ell_max)
    # powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    # powers0 =results0.get_cmb_power_spectra(pars, CMB_unit='muK')


    totCL = powers['total']*sigma_ratio
    unlensedCL = powers['unlensed_scalar']*sigma_ratio

    AllTT[i] = np.hstack([para5[i], totCL[:,0] ])
    AllEE[i] = np.hstack([para5[i], totCL[:,1] ])
    AllBB[i] = np.hstack([para5[i], totCL[:,2] ])
    AllTE[i] = np.hstack([para5[i], totCL[:,3] ])


    # np.save('../Cl_data/Data/LatintotCLP4'+str(totalFiles)+'_'+str(i) +'.npy', totCL)
    # np.save('../Cl_data/Data/LatinunlensedCLP4'+str(totalFiles)+'_'+str(i)+'.npy', unlensedCL)

SaveCls = True

if SaveCls:

    ls = np.arange(totCL.shape[0])

    # np.save('../Cl_data/Data/LatinPara5P4_'+str(totalFiles)+'.npy', para5)
    np.savetxt('../Cl_data/Data/Extended_ls_'+str(totalFiles)+'.txt', ls)

    np.savetxt('../Cl_data/Data/ExtendedTTCl_'+str(totalFiles)+'.txt', AllTT)
    np.savetxt('../Cl_data/Data/ExtendedEECl_'+str(totalFiles)+'.txt', AllEE)
    np.savetxt('../Cl_data/Data/ExtendedBBCl_'+str(totalFiles)+'.txt', AllBB)
    np.savetxt('../Cl_data/Data/ExtendedTECl_'+str(totalFiles)+'.txt', AllTE)

time1 = time.time()
print('camb time:', time1 - time0)

PlotCls = True

if PlotCls:

    MainDir = '../Cl_data/'
    PlotsDir = MainDir+'Plots/'+'ExtendedPlots/'
    paramNo = 9


    sortedArg = np.argsort(para5[:, paramNo])


    plt.figure(32)

    fig, ax = plt.subplots(2,2, figsize = (12,8))

    lineObj = ax[0,0].plot(AllTT[:, num_para + 1:].T[:, sortedArg])
    ax[0,0].set_yscale('log')
    ax[0,0].set_xscale('log')
    ax[0,0].set_ylabel(r'$C^{TT}_l$')
    ax[0,0].set_xlabel('$l$')

    ax[0,0].legend(iter(lineObj), para5[:, paramNo][sortedArg].round(decimals=2),
                   title = AllLabels[paramNo])

    # ax[0,0].legend(iter(lineObj), tau.round(decimals=2), title = r'\tau')
    # ax[0,0].legend(iter(lineObj), OmegaA.round(decimals=2), title = r'\omega_a')
    # ax[0,0].legend(iter(lineObj), mnu.round(decimals=2), title = r'$\sum m_\nu$')
    # ax[0,0].legend(iter(lineObj), para5[:, 2][sortedArg].round(decimals=4), title = r'$\sigma_8$')


    ax[1,0].plot(AllTE[:, num_para + 1:].T[:, sortedArg])
    ax[1,0].set_xscale('log')
    ax[1,0].set_ylabel(r'$C^{TE}_l$')
    ax[1,0].set_xlabel('$l$')

    ax[1,1].plot(AllEE[:, num_para + 1:].T[:, sortedArg])
    ax[1,1].set_yscale('log')
    ax[1,1].set_xscale('log')
    ax[1,1].set_ylabel(r'$C_l^{EE}$')
    ax[1,1].set_xlabel('$l$')


    ax[0,1].plot(AllBB[:, num_para + 1:].T[:, sortedArg])
    ax[0,1].set_ylabel(r'$C_l^{BB}$')
    # ax[0,1].set_yscale('log')
    ax[0,1].set_xscale('log')
    ax[0,1].set_xlabel('$l$')
    plt.savefig(PlotsDir + 'Param' + str(paramNo) + '_ExtendedClAll_'+str(totalFiles)+'.png')


    plt.show()
