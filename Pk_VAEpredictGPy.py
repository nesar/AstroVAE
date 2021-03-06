"""
GP fit for W matrix or latent z (encoded representation)

Uses George - package by Dan Foreman McKay - better integration with his MCMC package.
pip install george  - http://dan.iel.fm/george/current/user/quickstart/

"""
print(__doc__)

# Higdon et al 2008, 2012
# Check David's talk for plots of spectrum, and other things.

import numpy as np

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model

import paramsPk as params
#import Cl_load
import SetPub
SetPub.set_pub()



def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)



###################### PARAMETERS ##############################

original_dim = params.original_dim # 2549
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

import GPy

# ----------------------------- i/o ------------------------------------------


Trainfiles = np.loadtxt(DataDir + 'P'+str(num_para)+ClID+'_'+str(num_train)+'.txt')
Testfiles = np.loadtxt(DataDir + 'P'+str(num_para)+ClID+'_'+str(num_test)+'.txt')

Cl_Original = (Testfiles[:, num_para:])  # [2:3]



# para_train = Trainfiles[:, num_para+2:]
# para_test = Testfiles[:, num_para+2:]
para_train = Trainfiles[:, 0: num_para]
para_test =  Testfiles[:, 0: num_para]

print(para_train.shape, 'train sequences')
print(para_test.shape, 'test sequences')
# print(y_train.shape, 'train sequences')
# print(y_test.shape, 'test sequences')

ls = np.loadtxt( DataDir + 'P'+str(num_para)+'kh_'+str(num_train)+'.txt')[:]

#----------------------------------------------------------------------------

# meanFactor = np.min( [np.min(para_train), np.min(para_test ) ])
# print('-------mean factor:', meanFactor)
# para_train = para_train.astype('float32') - meanFactor #/ 255.
# para_test = para_test.astype('float32') - meanFactor #/ 255.
#

# para_train = np.log10(para_train) #para_train[:,2:] #
# para_test =  np.log10(para_test) #para_test[:,2:] #

# normFactor = np.max( [np.max(para_train), np.max(para_test ) ])

normFactor = np.loadtxt(DataDir+'normfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt')
meanFactor = np.loadtxt(DataDir+'meanfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt')

print('-------normalization factor:', normFactor)
print('-------rescaling factor:', meanFactor)


################# ARCHITECTURE ###############################



LoadModel = True
if LoadModel:

    # fileOut = 'VanillaModel_tot'+str(num_train)+'_batch'+str(batch_size)+'_lr'+str( learning_rate)+'_decay'+str(decay_rate)+'_z'+str(latent_dim)+'_epoch'+str(num_epochs)

    # vae = load_model(ModelDir + 'fullAE_' + fileOut + '.hdf5')
    encoder = load_model(ModelDir + 'EncoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    decoder = load_model(ModelDir + 'DecoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    history = np.loadtxt(ModelDir + 'TrainingHistoryP'+str(num_para)+ClID+'_'+fileOut+'.txt')







# # ------------------------------------------------------------------------------


# # ------------------------------------------------------------------------------
encoded_xtrain = np.loadtxt(DataDir + 'encoded_xtrainP'+str(num_para)+ClID+'_'+ fileOut +'.txt')
encoded_xtest = np.loadtxt(DataDir+'encoded_xtestP'+str(num_para)+ClID+'_'+ fileOut +'.txt')


# ------------------------------------------------------------------------------
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
# ------------------------------------------------------------------------------


max_relError = 0
ErrTh = 1.0
PlotRatio = True
IfVariance = False # Computation is a lot slower with Variance


W_predArray = np.zeros(shape=(num_test,latent_dim))
W_varArray = np.zeros(shape=(num_test,latent_dim))

x_decoded = np.zeros(shape=(num_test,original_dim))
x_decodedmax = np.zeros(shape=(num_test,original_dim))
x_decodedmin = np.zeros(shape=(num_test,original_dim))


# k = GPy.kern.Matern52(1, .3)
# K=GPy.kern.RBF(1)



if PlotRatio:

    # W_pred = np.array([np.zeros(shape=latent_dim)])
    # W_pred_var = np.array([np.zeros(shape=latent_dim)])


    # kern = GPy.kern.Matern52(input_dim= num_para)
    kern = GPy.kern.Matern52(5, 0.1)


    if (IfVariance == False):

        #########################################################################################
        ## All GP fitting together -- works fine, except we get one value of variance for all
        # output dimensions, since they're considered independent


        m1 = GPy.models.GPRegression(para_train, encoded_xtrain, kernel=kern)
        m1.Gaussian_noise.variance.constrain_fixed(1e-16)
        m1.optimize(messages=True)

        GPmodelOutfile = DataDir + 'GPy_model'+ str(latent_dim)+ClID + fileOut

        m1.save_model( GPmodelOutfile, compress=True, save_data=True)


        m1 = GPy.models.GPRegression.load_model(GPmodelOutfile + '.zip')


        m1p = m1.predict(para_test)  # [0] is the mean and [1] the predictive
        W_predArray = m1p[0]
        W_varArray = m1p[1]


        np.savetxt(DataDir + 'WPredArray_GPyNoVariance'+ str(latent_dim)+ClID + '.txt', W_predArray)

        #########################################################################################

    else:  # With Variance run ---> Expensive

        for i in range(para_test.shape[0]):
        # for i in range(2):

            para_test_point = para_test[i].reshape(num_para, -1).T
            m = {}

            for j in range(latent_dim):

                print '========= GP fit run -- test case:', i, ' output dim:', j, '========'
                print


                m["fit{0}".format(j)] = GPy.models.GPRegression(para_train, encoded_xtrain[:, j].reshape(
                    encoded_xtrain.shape[0], -1), kernel=kern)
                m["fit{0}".format(j)].Gaussian_noise.variance.constrain_fixed(1e-12)
                m["fit{0}".format(j)].optimize(messages=True)



                W_predArray[i, j], W_varArray[i, j] = m["fit{0}".format(j)].predict(para_test_point)



        np.savetxt(DataDir + 'WPredArray_GPy'+ str(latent_dim) +ClID+'.txt', W_predArray)
        np.savetxt(DataDir + 'WvarArray_GPy'+ str(latent_dim) +ClID+ '.txt', W_varArray)


#### ------------------------ Can be analyzed separately too ----------------------- ###
### LOAD GO MODEL AND EMULATE




plt.figure(992, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom = 0.15)  # set the spacing between axes.
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$P(k)$')

# ax0.set_title( r'$\text{' +fileOut + '}$')

ax1.axhline(y= 0, ls='dotted')
ax1.axhline(y=.01, ls='dashed')
ax1.axhline(y=-0.01, ls='dashed')


ax1.set_xscale('log')


ax0.set_yscale('log')
ax0.set_xscale('log')


ax1.set_xlabel(r'k Mpc/h')
ax1.set_ylabel(r'$P(k)^{emu}$/$P(k)^{camb} - 1$')




m1 = GPy.models.GPRegression.load_model(GPmodelOutfile + '.zip')


for i in range(para_test.shape[0]):
# for i in range(1):

        m1p = m1.predict(para_test[i].reshape(num_para, -1).T)  # [0] is the mean and [1] the predictive
        W_pred = m1p[0]
        x_decoded[i] = decoder.predict(  W_pred.reshape(latent_dim, -1).T   )




        # x_decoded[i] = decoder.predict(  W_predArray[i].reshape(latent_dim, -1).T   )#  BULK
# PREDICT

        # x_decodedmax[i] = decoder.predict(np.array( [W_predArray[i]] ) + np.sqrt(W_varArray[i]))
        # x_decodedmin[i] = decoder.predict(np.array( [W_predArray[i]] ) - np.sqrt(W_varArray[i]))



        ax0.plot(ls, (normFactor*x_decoded[i]) +meanFactor, 'r--', alpha= 0.5, lw = 1, label = 'emulated')
        ax0.plot(ls, (Cl_Original[i]), 'b--', alpha=0.5, lw = 1,  label = 'camb')

        cl_ratio = ( (normFactor * x_decoded[i]) +meanFactor)/ (Cl_Original[i])
        relError = ((cl_ratio) - 1)

        if (ClID == 'TE'):  # Absolute error instead (since TE gets crosses 0
            relError = ((normFactor * x_decoded[0]) + meanFactor - Cl_Original[i])




        ax0.plot(ls[np.abs(relError) > ErrTh], (normFactor*x_decoded[i] + meanFactor)[np.abs(
            relError) > ErrTh], 'gx', alpha=0.5, label= 'Err >'+str(ErrTh), markersize = '1')


        # ax1.plot(ls, ( (normFactor*x_decoded[i]) +meanFactor)/ (Cl_Original[i]), '-', lw = 0.5,
        #                  label = 'emu/camb')

        ax1.plot(ls, relError, 'k-', lw = 0.5, label = 'emu/camb', alpha = 0.9)


        print(i, 'ERR min max:', np.array([(relError).min(), (relError).max()]) )
        max_relError = np.max( [np.max(np.abs(relError)) , max_relError] )


plt.figure(992)
# plt.tight_layout()
plt.savefig(PlotsDir + 'TestGPy'+str(num_para)+ClID+fileOut+'.png')



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
    plt.savefig(PlotsDir + 'TrainingLoss_'+fileOut+'_relError'+ str( np.int(max_relError) ) +ClID+'.png')

#plt.show()

print(50*'#')
print(fileOut)
print('train loss: ', train_loss[-1])
print('test loss: ', val_loss[-1])
print
print('max rel error:', max_relError)
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


    inputArray = np.hstack([encoded_xtrain, encoded_xtrain])
    df = pd.DataFrame(inputArray, columns=AllLabels)
    axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'b')


    # df = pd.DataFrame(encoded_test_original, columns=AllLabels)
    # axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'b')
    # df = pd.DataFrame(  W_predArray, columns=AllLabels)
    # axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2, color = 'k')
    #plt.show()









# plt.show()


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
#    plt.plot(np.fft.rfft(para_train[i,:]))
#plt.xscale('log')

#plt.plot( np.fft.irfft( np.fft.rfft(para_train[0,:]) ), 'r' )
#plt.plot(para_train[0,:], 'b-.')



################3333333

# GP Doesn't seem to interpolate z space
# Even training points aren't mapped properly!!!
# Is the z space too weird?

# Same problem with my PCA+GP analysis
# Is the problems with epison_std being too small??



###---------------- Plot delta z vs delta para_train

## to check how well encoding does

delta_z = np.zeros(shape=para_train.shape[0] )
delta_xtrain = np.zeros(shape=para_train.shape[0] )

for i in range(encoded_xtrain.shape[1]):
    delta_z[i] = np.sqrt( np.mean(   (para_train[0] - para_train[i])**2  )  )
    delta_xtrain[i] = np.sqrt( np.mean(   (encoded_xtrain[0] -  encoded_xtrain[i])**2  )  )

delta_ztest = np.zeros(shape=para_test.shape[0] )
delta_xtest = np.zeros(shape=para_test.shape[0] )

for i in range(encoded_xtest.shape[0]):
    delta_ztest[i] = np.sqrt( np.mean(   (para_test[0] - para_test[i])**2  )  )
    delta_xtest[i] = np.sqrt( np.mean(   (encoded_xtest[0] -  encoded_xtest[i])**2  )  )



plt.figure(143, figsize=((5,5)))
plt.plot(delta_z, delta_xtrain, 'bx', markersize = 1, alpha = 0.5, label = 'train')
plt.plot(delta_ztest, delta_xtest, 'rx', markersize = 4, label = 'test')




plt.xlabel(r'$\bigtriangleup z$')
plt.ylabel(r'$\bigtriangleup$ $x$')

plt.legend()
plt.tight_layout()
plt.savefig(PlotsDir + 'SensitivityP'+str(num_para)+ClID+fileOut+'.png')
