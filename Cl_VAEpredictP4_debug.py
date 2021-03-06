"""
GP fit for W matrix or latent z (encoded representation)

Uses George - package by Dan Foreman McKay - better integration with his MCMC package.
pip install george  - http://dan.iel.fm/george/current/user/quickstart/

"""
print(__doc__)

# Higdon et al 2008, 2012
# Check David's talk for plots of spectrum, and other things.

import numpy as np

import matplotlib as mpl
# mpl.use('Agg')

import matplotlib.pyplot as plt

from keras.models import load_model

import params_debug as params
#import Cl_load
import SetPub
SetPub.set_pub()


plt.rc('text', usetex=False)   # Slower


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

# meanFactor = np.min( [np.min(x_train), np.min(x_test ) ])
# print('-------mean factor:', meanFactor)
# x_train = x_train.astype('float32') - meanFactor #/ 255.
# x_test = x_test.astype('float32') - meanFactor #/ 255.
#

# x_train = np.log10(x_train) #x_train[:,2:] #
# x_test =  np.log10(x_test) #x_test[:,2:] #

# normFactor = np.max( [np.max(x_train), np.max(x_test ) ])


###### universal rescaling ########
'''
normFactor = np.loadtxt(DataDir+'normfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt')
meanFactor = np.loadtxt(DataDir+'meanfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt')
'''

######### New l-dependant rescaling ###########
normFactor = np.loadtxt(DataDir+'normfactorPArr'+str(num_para)+ClID+'_'+ fileOut +'.txt')
meanFactor = np.loadtxt(DataDir+'meanfactorPArr'+str(num_para)+ClID+'_'+ fileOut +'.txt')


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

    # fileOut = 'VanillaModel_tot'+str(num_train)+'_batch'+str(batch_size)+'_lr'+str( learning_rate)+'_decay'+str(decay_rate)+'_z'+str(latent_dim)+'_epoch'+str(num_epochs)

    # vae = load_model(ModelDir + 'fullAE_' + fileOut + '.hdf5')
    encoder = load_model(ModelDir + 'EncoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    decoder = load_model(ModelDir + 'DecoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    history = np.loadtxt(ModelDir + 'TrainingHistoryP'+str(num_para)+ClID+'_'+fileOut+'.txt')

# vae = load_model(ModelDir + 'fullAEP'+str(num_para)+ClID+'_'+ fileOut + '.hdf5')


import george
from george.kernels import Matern32Kernel# , ConstantKernel, WhiteKernel, Matern52Kernel

# kernel = ConstantKernel(0.5, ndim=num_para) * Matern52Kernel(0.9, ndim=num_para) + WhiteKernel( 0.1, ndim=num_para)
#kernel = Matern32Kernel(1000, ndim=num_para)
# kernel = Matern32Kernel( [1000,2000,2000,1000,1000], ndim=num_para)
kernel = Matern32Kernel( [1000,4000,3000,1000,2000], ndim=num_para)
#kernel = Matern32Kernel( [1,0.5,1,1.4,0.5], ndim=num_para)

#kernel = Matern32Kernel(ndim=num_para)

# This kernel (and more importantly its subclasses) computes
# the distance between two samples in an arbitrary metric and applies a radial function to this distance.
# metric: The specification of the metric. This can be a float, in which case the metric is considered isotropic
# with the variance in each dimension given by the value of metric.
# Alternatively, metric can be a list of variances for each dimension. In this case, it should have length ndim.
# The fully general not axis aligned metric hasn't been implemented yet

# PLOTTING y_train and y_test

# import pandas as pd
# plt.figure(431)
# AllLabels = [r'$\Omega_m$', r'$\Omega_b$', r'$\sigma_8$', r'$h$', r'$n_s$']
# df = pd.DataFrame(y_train, columns=AllLabels)
# axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'b')
# df = pd.DataFrame(y_test, columns=AllLabels)
# axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'k')
# plt.tight_layout()


# plt.savefig('scatter_matrix.png')

# ------------------------------------------------------------------------------

# y_train1 = np.load(DataDir+'para5_'+str(num_train)+'.npy')








# maxP0 = np.max(np.append( y_train[:0], y_test[:,0]))
# maxP1 = np.max(np.append( y_train[:1], y_test[:,1]))
# maxP2 = np.max(np.append( y_train[:2], y_test[:,2]))
# maxP3 = np.max(np.append( y_train[:3], y_test[:,3]))
# maxP4 = np.max(np.append( y_train[:5], y_test[:,4]))
#
# minP0 = np.min(np.append( y_train[:0], y_test[:,0]))
# minP1 = np.min(np.append( y_train[:1], y_test[:,1]))
# minP2 = np.min(np.append( y_train[:2], y_test[:,2]))
# minP3 = np.min(np.append( y_train[:3], y_test[:,3]))
# minP4 = np.min(np.append( y_train[:5], y_test[:,4]))
#
#
#
# X1a = rescale01(minP0, maxP0, y_train[:, 0][:, np.newaxis])
# X2a = rescale01(minP1, maxP1, y_train[:, 1][:, np.newaxis])
# X3a = rescale01(minP2, maxP2, y_train[:, 2][:, np.newaxis])
# X4a = rescale01(minP3, maxP3, y_train[:, 3][:, np.newaxis])
# X5a = rescale01(minP4, maxP4, y_train[:, 4][:, np.newaxis])




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
    # ls = np.log10(np.load('../Cl_data/Data/Latinls_' + str(num_test) + '.npy')[2:])
    # PkOriginal = np.log10(np.load('../Cl_data/Data/LatinCl_'+str(num_test)+'.npy')[:,
    # 2:]) # Original

    # ls = np.loadtxt(DataDir + 'LatinlsP'+str(num_para)+'_' + str(num_test) + '.txt')[2:]#[2::2]

    # Cl_Original = np.load(DataDir + 'LatinCl_'+str(num_test)+'.npy')[:,2:]
    # RealParaArray = np.load(DataDir + 'LatinPara5_'+str(num_test)+'.npy')

    Cl_Original = (normFactor* x_test) + meanFactor  #[2:3]
    RealParaArray = y_test#[0:1]#[2:3]


    # Cl_Original = (normFactor*x_train)[0:10]
    # RealParaArray = y_train[0:10]



    for i in range(np.shape(RealParaArray)[0]):

        RealPara = RealParaArray[i]

        # RealPara[0] = rescale01(min, maxP0, RealPara[0])
        # RealPara[1] = rescale01(minP1, maxP1, RealPara[1])
        # RealPara[2] = rescale01(minP2, maxP2, RealPara[2])
        # RealPara[3] = rescale01(minP3, maxP3, RealPara[3])
        # RealPara[4] = rescale01(minP4, maxP4, RealPara[4])


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

        # plt.figure(94, figsize=(8,6))
        # plt.title('Autoencoder+GP fit')
        # # cl_ratio = 10**(normFactor*x_decoded[0])/10**(Cl_Original[i])
        #
        # cl_ratio = (normFactor*x_decoded[0])/(Cl_Original[i])
        #
        #
        # relError = 100*((cl_ratio) - 1)
        #
        # plt.plot(ls, cl_ratio, alpha=.8, lw = 1.0)
        # plt.ylim(0.95, 1.05)
        # # plt.xscale('log')
        # plt.xlabel(r'$l$')
        # plt.ylabel(r'$C_l^{GPAE}$/$C_l^{Original}$')
        # plt.title(fileOut)
        # # plt.legend()
        # plt.tight_layout()




        if i in PlotSampleID:


            ax0.plot(ls, (normFactor* x_decoded[0])+meanFactor  , 'r--', alpha= 0.8, lw = 1, \
                                                                                            label = 'emulated')
            ax0.plot(ls, (Cl_Original[i]), 'b--', alpha=0.8, lw = 1,  label = 'camb')

            cl_ratio = ((normFactor * x_decoded[0])+meanFactor) / Cl_Original[i]
            relError = ((cl_ratio) - 1)


            if (ClID == 'TE'):   # Absolute error instead (since TE gets crosses 0
                relError = ((normFactor * x_decoded[0]) + meanFactor - Cl_Original[i] )


            # cl_ratio = 2.*(((normFactor * x_decoded[0])+meanFactor) - Cl_Original[i])/\
            #            (((normFactor * x_decoded[0])+meanFactor) + Cl_Original[i])
            #
            # relError = 100 * ((cl_ratio) - 1)


            # ax0.plot(ls[np.abs(relError) > ErrTh], ( (normFactor*x_decoded[0])+meanFactor  )[np.abs(
            #     relError) > ErrTh], 'gx', alpha=0.7, label= 'Err >'+str(ErrTh), markersize = '1')


            ax1.plot(ls, relError, 'k-', lw = 0.25, label = 'emu/camb', alpha = 0.5)

            # plt.savefig(PlotsDir + 'TestGridP'+str(num_para)+''+fileOut+'.png')



            # plt.figure(99, figsize=(8,6))
            # plt.title('Autoencoder+GP fit')
            # # plt.plot(ls, normFactor * x_test[::].T, 'gray', alpha=0.1)
            #
            # # plt.plot(ls, 10**(normFactor*x_decoded[0]), 'r--', alpha= 0.5, lw = 1, label = 'emulated')
            # # plt.plot(ls, 10**(Cl_Original[i]), 'b--', alpha=0.5, lw = 1, label = 'original')
            #
            # plt.plot(ls, (normFactor*x_decoded[0]), 'r--', alpha= 0.8, lw = 1, label = 'emulated')
            # plt.plot(ls, (Cl_Original[i]), 'b--', alpha=0.8, lw = 1, label = 'original')
            #
            #
            # # plt.xscale('log')
            # plt.xlabel(r'$l$')
            # plt.ylabel(r'$C_l$')
            # plt.legend()
            # # plt.tight_layout()
            #
            # plt.plot(ls[np.abs(relError) > ErrTh], normFactor*x_decoded[0][np.abs(relError) >
            #                                                               ErrTh], 'gx',
            #          alpha=0.7, label='bad eggs', markersize = '1')
            # plt.title(fileOut)
            #
            # plt.savefig(PlotsDir + 'TestP'+str(num_para)+''+fileOut+'.png')
        #plt.show()
        print(i, 'ERR min max:', np.array([(relError).min(), (relError).max()]) )

        max_relError = np.max( [np.max(np.abs(relError)) , max_relError] )

    # plt.figure(94, figsize=(8,6))
    # plt.axhline(y=1, ls='-.', lw=1.5)
    # plt.savefig(PlotsDir + 'RatioP'+str(num_para)+''+fileOut+'.png')

    #plt.show()

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




# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


################### Sent to Mickeal  #---------------------------------------------


# Trainfiles = np.loadtxt(DataDir + 'P'+str(num_para)+'Cl_'+str(num_train)+'.txt')
# Testfiles = np.loadtxt(DataDir + 'P'+str(num_para)+'Cl_'+str(num_test)+'.txt')
#
# para5_train = Trainfiles[:, 0: num_para]
# para5_new =  Testfiles[:, 0: num_para]
#
# encoded_train = np.loadtxt(DataDir + 'encoded_xtrainP'+str(num_para)+'_'+ fileOut +'.txt')
# encoded_xtest_original = np.loadtxt(DataDir+'encoded_xtestP'+str(num_para)+'_'+ fileOut +'.txt')
#
# np.savetxt('para4_train.txt', para5_train)
# np.savetxt('para4_new.txt', para5_new)
#
# np.savetxt('encoded_trainP'+str(num_para)+'.txt', encoded_train)
# np.savetxt('encoded_test_originalP'+str(num_para)+'.txt', encoded_xtest_original)


#
#
# PlotParamsScatter = False
#
# if PlotParamsScatter:
#
#     plt.figure(433)
#
#     import pandas as pd
#
#     AllLabels = [r'$\Omega_m$', r'$\Omega_b$', r'$\sigma_8$', r'$h$', r'$n_s$']
#     inputArray = para5_new
#     df = pd.DataFrame(inputArray, columns=AllLabels)
#     pd.tools.plotting.scatter_matrix(df, alpha=0.8, color='b', diagonal='kde')
#
#     inputArray = para5_train
#     df = pd.DataFrame(inputArray, columns=AllLabels)
#     pd.tools.plotting.scatter_matrix(df, alpha=0.2, color='r', diagonal='kde')



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------



#
#
# plt.figure(1542)
# # plt.plot(encoded_train.T, 'b', alpha = 0.02 )
# plt.plot(encoded_test_original.T, 'r', alpha = 0.2)
# plt.plot(W_predArray.T, 'k--', alpha = 0.3)
# plt.show()

#
# plt.figure(423)
# plt.scatter(encoded_xtest_original[:,0], encoded_xtest_original[:,1], c = para5_new[:,0],
#             cmap=plt.cm.afmhot, s = 10)
# plt.scatter(encoded_train[:,0], encoded_train[:,1], c = para5_train[:,0], s = 5, alpha = 0.3)
#
# plt.show()

##---------------------------------------------


#
#
# plt.figure(5343)
# plt.plot(W_predArray/encoded_xtest_original, 'ko', markersize=2)
# plt.yscale('symlog')
#
#
# # plt.figure(5343)
# indCheck = 6
# plt.scatter(W_predArray[:,indCheck], encoded_xtest_original[:,indCheck], c = para5_new[:,0])


PlotScatter = False
if PlotScatter:
    plt.figure(431)
    import pandas as pd

    # AllLabels = []

    # for ind in np.arange(1, num_para+1):
    #     AllLabels.append(str("v{0}".format(ind)))

    AllLabels = [r'$\Omega_m$', r'$\Omega_b$', r'$\sigma_8$', r'$h$', r'$n_s$']


    for ind in np.arange(1, latent_dim+1):
        AllLabels.append(str("z{0}".format(ind)))


    inputArray = np.hstack([y_train, y.T])
    df = pd.DataFrame(inputArray, columns=AllLabels)
    axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'b')


    # df = pd.DataFrame(encoded_test_original, columns=AllLabels)
    # axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'b')
    # df = pd.DataFrame(  W_predArray, columns=AllLabels)
    # axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'k')
    #plt.show()




# ------------------------------------------------------------------------------

# Cosmic Variance emulator


# Next step -- uncertainty quantification -- check Laurence et al
# Parameter estimation  -- mcmc or variational bayes
# Go back to check P(k) -- we may just use the simulation suite data.
# Generate better data sampling - latin hc?
#
# Deeper network!
# adding hidden layers seems to be fixed -- check again

# Talk to Jason
# CAMB install gfortran issue in Phoenix
# Generate lots of outputs
# training check using 32 files.
# Why emulate when there's a theoretical model?

# Still haven't figured how to change sigma_8 in CAMB

# CAMB - linear theory -- good for demonstration
# Future -- use simulations for a proper cmb power spectrum

# Change loss function and see if it changes anything
## can loss include something from GP?

# combine a few files -- one for io, one for training (AE+GP), one for testing?

## encoded arrays are all same!!!

## Tried avg, didn't help
## Try fft ??


## Plot GP functions cuz people want to look at it

## Hyper-parameter optimization softwares

#plt.figure(35)
#for i in range(10):
#    plt.plot(np.fft.rfft(x_train[i,:]))
#plt.xscale('log')

#plt.plot( np.fft.irfft( np.fft.rfft(x_train[0,:]) ), 'r' )
#plt.plot(x_train[0,:], 'b-.')



################3333333

# GP Doesn't seem to interpolate z space
# Even training points aren't mapped properly!!!
# Is the z space too weird?

# Same problem with my PCA+GP analysis
# Is the problems with epison_std being too small??



###---------------- Plot delta z vs delta x_train

## to check how well encoding does

delta_z = np.zeros(shape=y.shape[1] )
delta_xtrain = np.zeros(shape=y.shape[1] )

for i in range(y.shape[1]):
    delta_z[i] = np.sqrt( np.mean(   (y.T[0] - y.T[i])**2  )  )
    delta_xtrain[i] = np.sqrt( np.mean(   (x_train[0] -  x_train[i])**2  )  )

delta_ztest = np.zeros(shape=encoded_xtest_original.shape[0] )
delta_xtest = np.zeros(shape=encoded_xtest_original.shape[0] )

for i in range(encoded_xtest_original.shape[0]):
    delta_ztest[i] = np.sqrt( np.mean(   (encoded_xtest_original[0] - encoded_xtest_original[i])**2  )  )
    delta_xtest[i] = np.sqrt( np.mean(   (x_test[0] -  x_test[i])**2  )  )



plt.figure(143, figsize=((5,5)))
plt.plot(delta_z, delta_xtrain, 'bx', markersize = 1, alpha = 0.5, label = 'train')
plt.plot(delta_ztest, delta_xtest, 'rx', markersize = 4, label = 'test')




plt.xlabel(r'$\bigtriangleup z$')
plt.ylabel(r'$\bigtriangleup$ $x$')

plt.legend()
plt.tight_layout()
plt.savefig(PlotsDir + 'SensitivityP'+str(num_para)+ClID+fileOut+'.png')


plt.show()
