"""
sudo pip install camb

from:
http://camb.readthedocs.io/en/latest/CAMBdemo.html


"""

import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
print('Using CAMB installed at %s'%(os.path.realpath(os.path.join(os.getcwd(),'..'))))
#uncomment this if you are running remotely and want to keep in synch with repo changes
#if platform.system()!='Windows':
#    !cd $HOME/git/camb; git pull github master; git log -1
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model, initialpower
import itertools
print('CAMB version: %s '%camb.__version__)
import SetPub
SetPub.set_pub()


#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);

#calculate results for these parameters
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)

#plot the total lensed CMB power spectra versus unlensed, and fractional difference
totCL=powers['total']
unlensedCL=powers['unlensed_scalar']
print(totCL.shape)
#Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
#The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
ls = np.arange(totCL.shape[0])
fig, ax = plt.subplots(2,2, figsize = (12,12))
ax[0,0].plot(ls,totCL[:,0], color='k')
ax[0,0].plot(ls,unlensedCL[:,0], color='r')
ax[0,0].set_title('TT')
ax[0,1].plot(ls[2:], 1-unlensedCL[2:,0]/totCL[2:,0]);
ax[0,1].set_title(r'$\Delta TT$')
ax[1,0].plot(ls,totCL[:,1], color='k')
ax[1,0].plot(ls,unlensedCL[:,1], color='r')
ax[1,0].set_title(r'$EE$')
ax[1,1].plot(ls,totCL[:,3], color='k')
ax[1,1].plot(ls,unlensedCL[:,3], color='r')
ax[1,1].set_title(r'$TE$');
for ax in ax.reshape(-1): ax.set_xlim([2,2500]);


#-------------------------------------------------------------------------------
print(pars) # Shows all parameters
# -------------------------------------------------------------------------------








nsize = 2

OmegaM = np.linspace(0.12, 0.155, nsize)
# OmegaM = normalGenerate(0.12, 0.155, nsize)  # Not sure if the data should be generated likewise?
Omegab = np.linspace(0.0215, 0.0235, nsize)
sigma8 = np.linspace(0.7, 0.9,nsize)
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

    plt.figure(10)
    # plt.plot(ls,totCL[:,0], color='k', alpha = 0.4)
    plt.plot(ls,unlensedCL[:,0], color='r', alpha = 0.1)

plt.title('TT')
# plt.xscale('log')
plt.xlabel(r'$l(l+1)$')
plt.ylabel(r'$C_l$')

plt.show()