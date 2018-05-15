"""
For testing decoder output
input encoded_xtest from trained GP models

"""
print(__doc__)

# Higdon et al 2008, 2012
# Check David's talk for plots of spectrum, and other things.

import numpy as np

# import matplotlib as mpl
# mpl.use('Agg')
# import SetPub
# SetPub.set_pub()
import matplotlib.pyplot as plt

from keras.models import load_model

import params
#import Cl_load



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




Trainfiles = np.loadtxt(DataDir + 'P'+str(num_para)+ClID+'Cl_'+str(num_train)+'.txt')
Testfiles = np.loadtxt(DataDir + 'P'+str(num_para)+ClID+'Cl_'+str(num_test)+'.txt')

# Cl_Original = (Testfiles[:, num_para+2:])  # [2:3]



# para_train = Trainfiles[:, num_para+2:]
# para_test = Testfiles[:, num_para+2:]
para_train = Trainfiles[:, 0: num_para]
para_test =  Testfiles[:, 0: num_para]

print(para_train.shape, 'train sequences')
print(para_test.shape, 'test sequences')
# print(y_train.shape, 'train sequences')
# print(y_test.shape, 'test sequences')

ls = np.loadtxt( DataDir + 'P'+str(num_para)+'ls_'+str(num_train)+'.txt')[2:]

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






# Trainfiles = np.loadtxt(DataDir + 'P'+str(num_para)+'Cl_'+str(num_train)+'.txt')
# Testfiles = np.loadtxt(DataDir + 'P'+str(num_para)+'Cl_'+str(num_test)+'.txt')

# x_train = Trainfiles[:, num_para+2:]
x_test = Testfiles[:, num_para+2:]
y_train = Trainfiles[:, 0: num_para]
y_test =  Testfiles[:, 0: num_para]

# print(x_train.shape, 'train sequences')
# print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')

ls = np.loadtxt( DataDir + 'P'+str(num_para)+'ls_'+str(num_train)+'.txt')[2:]

#----------------------------------------------------------------------------

# normFactor = np.loadtxt(DataDir+'normfactorP'+str(num_para)+'_'+ fileOut +'.txt')
# print('-------normalization factor:', normFactor)

# x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.


# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# ------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------
# encoded_xtrain = np.loadtxt(DataDir + 'encoded_xtrainP'+str(num_para)+ClID+'_'+ fileOut +'.txt')
# encoded_xtest = np.loadtxt(DataDir+'encoded_xtestP'+str(num_para)+ClID+'_'+ fileOut +'.txt')


# y = np.loadtxt(DataDir + 'encoded_xtrainP'+str(num_para)+'_'+ fileOut +'.txt').T
# encoded_xtest_original = np.loadtxt(DataDir+'encoded_xtestP'+str(num_para)+'_'+ fileOut +'.txt')

# ------------------------------------------------------------------------------
# np.set_printoptions(precision=3)
# np.set_printoptions(suppress=True)
# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# ------------------------------------------------------------------------------


################# ARCHITECTURE ###############################


LoadModel = True
if LoadModel:

    # fileOut = 'VanillaModel_tot'+str(num_train)+'_batch'+str(batch_size)+'_lr'+str( learning_rate)+'_decay'+str(decay_rate)+'_z'+str(latent_dim)+'_epoch'+str(num_epochs)

    # vae = load_model(ModelDir + 'fullAE_' + fileOut + '.hdf5')
    encoder = load_model(ModelDir + 'EncoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    decoder = load_model(ModelDir + 'DecoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    history = np.loadtxt(ModelDir + 'TrainingHistoryP'+str(num_para)+ClID+'_'+fileOut+'.txt')








plt.figure(999, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom = 0.15)  # set the spacing between axes.
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$C_l$')
# ax0.set_title( r'$\text{' +fileOut + '}$')

ax1.axhline(y=1, ls='dotted')
ax1.axhline(y=1.01, ls='dashed')
ax1.axhline(y=0.99, ls='dashed')

ax1.set_xlabel(r'$l$')

ax1.set_ylabel(r'$C_l^{emu}$/$C_l^{camb}$')
# ax1.set_ylim(0.976, 1.024)


PlotSampleID = np.array([ 0, 3, 4, 5, 11, 12])
#PlotSampleID = np.arange(x_test.shape[0])

max_relError = 0
ErrTh = 0.5
PlotRatio = True

# W_predArray = np.load('encoded_test_GP.npy')  ## From Mickael
# W_varArray = (np.load('Var_preds.npy')).T  ## From Mickael


# W_predArray = np.loadtxt(DataDir + 'W2PredArray_GPy'+ str(latent_dim) + '.txt')
# W_varArray = np.loadtxt(DataDir + 'W2varArray_GPy'+ str(latent_dim) + '.txt')

W_predArray = np.loadtxt(DataDir + 'WPredArray_GPy' + str(latent_dim) + ClID + '.txt')
W_varArray = np.loadtxt(DataDir + 'WvarArray_GPy' + str(latent_dim) + ClID + '.txt')

nsigma = 5.


if PlotRatio:


    Cl_Original = (normFactor*x_test)#[2:3]
    RealParaArray = y_test#[2:3]



    # Cl_Original = (normFactor*x_train)[0:10]
    # RealParaArray = y_train[0:10]



    for i in range(np.shape(RealParaArray)[0]):

        # RealPara = RealParaArray[i]


        W_pred = np.array([W_predArray[i]])
        # x_decoded = decoder.predict(W_pred*ymax)# + meanFactor
        x_decoded = decoder.predict(W_pred)# + meanFactor


        W_predmax = np.array( [W_predArray[i]] ) + nsigma*np.sqrt(W_varArray[i])
        # x_decoded = decoder.predict(W_pred*ymax)# + meanFactor
        x_decodedmax = decoder.predict(W_predmax)# + meanFactor

        W_predmin = np.array( [W_predArray[i]] ) - nsigma*np.sqrt(W_varArray[i])
        # x_decoded = decoder.predict(W_pred*ymax)# + meanFactor
        x_decodedmin = decoder.predict(W_predmin)# + meanFactor






        if i in PlotSampleID:



            x_decoded_fill = np.vstack( [x_decodedmin[0], x_decodedmax[0]] )

            x_decoded_lower = np.min(x_decoded_fill, axis=0)
            x_decoded_upper = np.max(x_decoded_fill, axis=0)

            ax0.fill_between(ls,  (normFactor*x_decoded_lower) + meanFactor ,
                             (normFactor*x_decoded_upper) + meanFactor,
                             alpha = 1.0, linewidth=1, linestyle='dashdot', facecolor='red')




            ax0.plot(ls, (normFactor*x_decoded[0]) + meanFactor, 'r-', alpha= 0.4, lw = 1,
                     label = 'emulated')
            ax0.plot(ls, Cl_Original[i], 'b--', alpha=0.5, lw = 1,  label = 'camb'  )

            cl_ratio = ((normFactor * x_decoded[0]) + meanFactor) / (Cl_Original[i])

            relError = 100.0*((cl_ratio) - 1)

            ax0.plot(ls[np.abs(relError) > ErrTh], normFactor*x_decoded[0][np.abs(relError) >
            ErrTh], 'gx', alpha=0.2, label= 'Err >'+str(ErrTh), markersize = '1')


            ax1.plot(ls, (  (normFactor*x_decoded[0])  + meanFactor) / (Cl_Original[i]), 'k-',
                     lw = 0.2, label = 'emu/camb')

            ax1.fill_between(ls, ( (normFactor*x_decoded_lower) + meanFactor)/ (Cl_Original[i]),
                             ( (normFactor*x_decoded_upper) + meanFactor) / (Cl_Original[i]),
                             alpha = 0.2,
                             facecolor = 'red')



            # ax0.plot(ls[np.abs(relError) > ErrTh], normFactor*x_decoded[0][np.abs(relError) >
            #         ErrTh], 'gx', alpha=0.5, label='bad eggs', markersize = '1')

            # plt.savefig(PlotsDir + 'TestP'+str(num_para)+''+fileOut+'.png')
        #plt.show()
        print(i, 'ERR min max:', np.array([(relError).min(), (relError).max()]) )

        max_relError = np.max( [np.max(np.abs(relError)) , max_relError] )

    # plt.figure(94, figsize=(8,6))
    # plt.axhline(y=1, ls='-.', lw=1.5)
    # plt.savefig(PlotsDir + 'RatioP'+str(num_para)+''+fileOut+'.png')

    #plt.show()
print(50*'-')
print('file:', fileOut)
# ------------------------------------------------------------------------------


plotLoss = False
if plotLoss:

    epochs =  history[0,:]
    train_loss = history[1,:]
    val_loss = history[2,:]


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
    plt.tight_layout()
    plt.savefig(PlotsDir + 'TrainingLoss_'+fileOut+'_relError'+ str( np.int(max_relError) ) +'.png')

plt.show()


print('max rel error', max_relError)
# print('train loss: ', train_loss[-1])
