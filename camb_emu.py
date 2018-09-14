"""
CAMB vs EMU comparision
"""
print(__doc__)

# Higdon et al 2008, 2012
# Check David's talk for plots of spectrum, and other things.

import numpy as np

import matplotlib as mpl
# mpl.use('Agg')

import matplotlib.pyplot as plt

from keras.models import load_model

import params
#import Cl_load
import SetPub
SetPub.set_pub()



def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)



###################### PARAMETERS ##############################

#original_dim = params.original_dim # 2549
#intermediate_dim2 = params.intermediate_dim2 # 1024
#intermediate_dim1 = params.intermediate_dim1 # 512
#intermediate_dim = params.intermediate_dim # 256
latent_dim = params.latent_dim # 10


ClID = params.ClID
num_train = params.num_train # 512
num_test = params.num_test # 32
num_para = params.num_para # 5

batch_size = params.batch_size # 8
num_epochs = params.num_epochs # 100
epsilon_mean = params.epsilon_mean # 1.0
epsilon_std = params.epsilon_std # 1.0
learning_rate = params.learning_rate # 1e-3
decay_rate = params.decay_rate # 0.0

noise_factor = params.noise_factor # 0.00

######################## I/O ##################################

DataDir = params.DataDir
PlotsDir = params.PlotsDir
ModelDir = params.ModelDir

fileOut = params.fileOut


# ----------------------------- i/o ------------------------------------------



Trainfiles = np.loadtxt(DataDir + 'P'+str(num_para)+ClID+'Cl_'+str(num_train)+'.txt')
Testfiles = np.loadtxt(DataDir + 'P'+str(num_para)+ClID+'Cl_'+str(num_test)+'.txt')


x_train = Trainfiles[:, num_para+2:]
x_test = Testfiles[:, num_para+2:]
y_train = Trainfiles[:, 0: num_para]
y_test =  Testfiles[:, 0: num_para]

print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')

ls = np.loadtxt( DataDir + 'P'+str(num_para)+'ls_'+str(num_train)+'.txt')[2:]

#----------------------------------------------------------------------------


normFactor = np.loadtxt(DataDir+'normfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt')
meanFactor = np.loadtxt(DataDir+'meanfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt')

print('-------normalization factor:', normFactor)
print('-------rescaling factor:', meanFactor)


x_train = x_train - meanFactor #/ 255.
x_test = x_test - meanFactor #/ 255.


x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# ------------------------------------------------------------------------------


################# ARCHITECTURE ###############################



LoadModel = True
if LoadModel:


    encoder = load_model(ModelDir + 'EncoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    decoder = load_model(ModelDir + 'DecoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    history = np.loadtxt(ModelDir + 'TrainingHistoryP'+str(num_para)+ClID+'_'+fileOut+'.txt')


import george
from george.kernels import Matern32Kernel# , ConstantKernel, WhiteKernel, Matern52Kernel


kernel = Matern32Kernel( [1000,4000,3000,1000,2000], ndim=num_para)


X1 = y_train[:, 0][:, np.newaxis]
X1a = rescale01(np.min(X1), np.max(X1), X1)

X2 = y_train[:, 1][:, np.newaxis]
X2a = rescale01(np.min(X2), np.max(X2), X2)

X3 = y_train[:, 2][:, np.newaxis]
X3a = rescale01(np.min(X3), np.max(X3), X3)

X4 = y_train[:, 3][:, np.newaxis]
X4a = rescale01(np.min(X4), np.max(X4), X4)

X5 = y_train[:, 4][:, np.newaxis]
X5a = rescale01(np.min(X5), np.max(X5), X5)


XY = np.array(np.array([X1a, X2a, X3a, X4a, X5a])[:, :, 0])[:, np.newaxis]



minmax_rescale_ytrain =  [[np.min(X1), np.max(X1)], [np.min(X2), np.max(X2)],
[np.min(X3), np.max(X3)],
[np.min(X4), np.max(X4)],
[np.min(X5), np.max(X5)] ]


np.savetxt(DataDir+'rescale_ytrain'+str(num_train)+ClID+'.txt', minmax_rescale_ytrain)



# # ------------------------------------------------------------------------------
y = np.loadtxt(DataDir + 'encoded_xtrainP'+str(num_para)+ClID+'_'+ fileOut +'.txt').T
encoded_xtest_original = np.loadtxt(DataDir+'encoded_xtestP'+str(num_para)+ClID+'_'+ fileOut +'.txt')

# ------------------------------------------------------------------------------
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
# ------------------------------------------------------------------------------






plt.figure(997, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom = 0.15)  # set the spacing between axes.
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$l(l+1)C_l/2\pi [\mu K^2]$')
# ax0.set_title( r'$\text{' +fileOut + '}$')
ax0.text(0.95, 0.95,ClID, horizontalalignment='center', verticalalignment='center',
         transform = ax0.transAxes, fontsize = 20)



ax1.axhline(y= 0, ls='dotted')
ax1.axhline(y=.01, ls='dashed')
ax1.axhline(y=-0.01, ls='dashed')
# ax1.set_ylim(0.976, 1.024)
# ax1.set_yscale('log')



ax1.set_xlabel(r'$l$')
ax1.set_ylabel(r'$C_l^{emu}$/$C_l^{camb} - 1$')
if (ClID == 'TE'): ax1.set_ylabel(r'$C_l^{emu}$ - $C_l^{camb}$')




# PlotSampleID = [6, 4, 23, 26, 17, 12, 30, 4]
PlotSampleID = np.arange(y_test.shape[0])[::]


max_relError = 0
ErrTh = 1.0
PlotRatio = True

W_predArray = np.zeros(shape=(num_test,latent_dim))

AllPred = np.zeros_like(x_test)

if PlotRatio:


    Cl_Original = (normFactor* x_test) + meanFactor  #[2:3]
    RealParaArray = y_test#[0:1]#[2:3]


    for i in range(np.shape(RealParaArray)[0]):

        RealPara = RealParaArray[i]

        RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
        RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
        RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2])
        RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
        RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])

        test_pts = RealPara[:num_para].reshape(num_para, -1).T

        # ------------------------------------------------------------------------------
        # y = np.load('../Pk_data/SVDvsVAE/encoded_xtrain.npy').T

        W_pred = np.array([np.zeros(shape=latent_dim)])
        W_pred_var = np.array([np.zeros(shape=latent_dim)])


        gp = {}
        for j in range(latent_dim):
            gp["fit{0}".format(j)] = george.GP(kernel)

            # gp["fit{0}".format(j)].optimize( XY[:,0,:].T, y[j], verbose=True)
            ###### optimizes parameters, takes longer

            gp["fit{0}".format(j)].compute(XY[:, 0, :].T)
            W_pred[:, j], W_pred_var[:, j] = gp["fit{0}".format(j)].predict(y[j], test_pts)#[0]
            print(20*'-', W_pred[:, j], W_pred_var[:, j])
            # W_pred_var[:, j] = gp["fit{0}".format(j)].predict(y[j], test_pts)[0]

        # ------------------------------------------------------------------------------

        W_predArray[i] = W_pred
        # x_decoded = decoder.predict(W_pred*ymax)# + meanFactor
        x_decoded = decoder.predict(W_pred)# + meanFactor

        AllPred[i] = (normFactor * x_decoded[0])+meanFactor


        if i in PlotSampleID:


            ax0.plot(ls, (normFactor* x_decoded[0])+meanFactor  , 'r--', alpha= 0.8, lw = 1, \
                                                                                            label = 'emulated')
            ax0.plot(ls, (Cl_Original[i]), 'b--', alpha=0.8, lw = 1,  label = 'camb')

            cl_ratio = ((normFactor * x_decoded[0])+meanFactor) / Cl_Original[i]
            relError = ((cl_ratio) - 1)


            if (ClID == 'TE'):   # Absolute error instead (since TE gets crosses 0
                relError = ((normFactor * x_decoded[0]) + meanFactor - Cl_Original[i] )



            ax1.plot(ls, relError, 'k-', lw = 0.25, label = 'emu/camb', alpha = 0.5)


        print(i, 'ERR min max:', np.array([(relError).min(), (relError).max()]) )

        max_relError = np.max( [np.max(np.abs(relError)) , max_relError] )


plt.figure(997)
# plt.tight_layout()
plt.savefig(PlotsDir + 'TestGridP'+str(num_para)+ClID+fileOut+'.png')



print(50*'-')
print('file:', fileOut)
# ------------------------------------------------------------------------------


epochs = history[0, :]
train_loss = history[1, :]
val_loss = history[2, :]

plotLoss = False
if plotLoss:

    plt.figure(867)
    fig, ax = plt.subplots(1,1, sharex= True, figsize = (8,6))
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epochs,train_loss, '-', lw =1.5)
    ax.plot(epochs,val_loss, '-', lw = 1.5)
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train loss','val loss'])
    #plt.text(5.75, 0.15, 'MaxRelError: %d'%np.int(max_relError) , fontsize=15)
    plt.title(fileOut)
    # plt.tight_layout()
    plt.savefig(PlotsDir + 'TrainingLoss_'+fileOut+ClID+'_relError'+ str( np.int(max_relError) ) +'.png')

#plt.show()

print(50*'#')
print(fileOut)
print(ClID)
print('train loss: ', train_loss[-1])
print('test loss: ', val_loss[-1])
print
print('max rel error:', 100*max_relError, 'percent')
print(50*'#')




import camb
from camb import model, initialpower

import time
time0 = time.time()

"""
CAMBFast/CLASS to be added next
CosmoMC works well with CAMB
http://camb.readthedocs.io/en/latest/CAMBdemo.html
https://wiki.cosmos.esa.int/planckpla2015/index.php/CMB_spectrum_%26_Likelihood_Code
"""

numpara = 5
# ndim = 2551
totalFiles =  8
lmax0 = 2500



para5 = np.loadtxt('../Cl_data/Data/LatinCosmoP5'+str(totalFiles)+'.txt')

# print(para5)

f, a = plt.subplots(5, 5, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.rcParams.update({'font.size': 8})


AllLabels = [r'$\tilde{\Omega}_m$', r'$\tilde{\Omega}_b$', r'$\tilde{\sigma}_8$', r'$\tilde{h}$',
             r'$\tilde{n}_s$']


#---------------------------------------
AllTT = np.zeros(shape=(totalFiles, numpara + lmax0 + 1) ) # TT
AllEE = np.zeros(shape=(totalFiles, numpara + lmax0 + 1) ) #
AllBB = np.zeros(shape=(totalFiles, numpara + lmax0 + 1) )
AllTE = np.zeros(shape=(totalFiles, numpara + lmax0 + 1) ) # Check if this is actually TE --
# negative
# values and CAMB documentation incorrect.

for i in range(totalFiles):
    print(i, para5[i])

    pars = camb.CAMBparams()

    pars.set_cosmology(H0=100*para5[i, 3], ombh2=para5[i, 1], omch2=para5[i, 0], mnu=0.06, omk=0,
                       tau=0.06)


    pars.InitPower.set_params(ns=para5[i, 4], r=0)


    pars.set_for_lmax(lmax= lmax0, max_eta_k=None, k_eta_fac=12.5 , lens_potential_accuracy=1,
                      lens_k_eta_reference = 20000)
    ## THIS IS ONLY FOR ACCURACY,
    ## actual lmax is set in results.get_cmb_power_spectra

    # model.lmax_lensed = 10000  ## doesn't work

    print model.lmax_lensed


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
    # pars.set_for_lmax(lmax= lmax0, k_eta_fac=2.5 , lens_potential_accuracy=0)  ## THIS IS ONLY


    #calculate results for these parameters
    results = camb.get_results(pars)


    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax0)

    totCL = powers['total']*r
    unlensedCL = powers['unlensed_scalar']*r

    AllTT[i] = np.hstack([para5[i], totCL[:,0] ])
    AllEE[i] = np.hstack([para5[i], totCL[:,1] ])
    AllBB[i] = np.hstack([para5[i], totCL[:,2] ])
    AllTE[i] = np.hstack([para5[i], totCL[:,3] ])


ls = np.arange(totCL.shape[0])

time1 = time.time()
print('camb time:', time1 - time0)


plt.figure(32)
plt.plot(AllTT[:, 7:].T)
plt.yscale('log')
plt.xscale('log')
plt.show()


