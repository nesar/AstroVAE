"""
For testing models from Mickael

"""
print(__doc__)

# Higdon et al 2008, 2012
# Check David's talk for plots of spectrum, and other things.

import numpy as np

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model

import params
#import Cl_load
#import SetPub
#SetPub.set_pub()


###################### PARAMETERS ##############################

#original_dim = params.original_dim # 2549
#intermediate_dim2 = params.intermediate_dim2 # 1024
#intermediate_dim1 = params.intermediate_dim1 # 512
#intermediate_dim = params.intermediate_dim # 256
latent_dim = params.latent_dim # 10

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


################# ARCHITECTURE ###############################


LoadModel = True
if LoadModel:

    # fileOut = 'VanillaModel_tot'+str(num_train)+'_batch'+str(batch_size)+'_lr'+str( learning_rate)+'_decay'+str(decay_rate)+'_z'+str(latent_dim)+'_epoch'+str(num_epochs)

    # vae = load_model(ModelDir + 'fullAE_' + fileOut + '.hdf5')
    encoder = load_model(ModelDir + 'EncoderP'+str(num_para)+'_' + fileOut + '.hdf5')
    decoder = load_model(ModelDir + 'DecoderP'+str(num_para)+'_' + fileOut + '.hdf5')
    history = np.loadtxt(ModelDir + 'TrainingHistoryP'+str(num_para)+'_'+fileOut+'.txt')



Trainfiles = np.loadtxt(DataDir + 'P'+str(num_para)+'Cl_'+str(num_train)+'.txt')
Testfiles = np.loadtxt(DataDir + 'P'+str(num_para)+'Cl_'+str(num_test)+'.txt')

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

normFactor = np.loadtxt(DataDir+'normfactorP'+str(num_para)+'_'+ fileOut +'.txt')
print('-------normalization factor:', normFactor)

x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# ------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------
y = np.loadtxt(DataDir + 'encoded_xtrainP'+str(num_para)+'_'+ fileOut +'.txt').T
encoded_xtest_original = np.loadtxt(DataDir+'encoded_xtestP'+str(num_para)+'_'+ fileOut +'.txt')

# ------------------------------------------------------------------------------
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# ------------------------------------------------------------------------------

# PlotSampleID = [6, 4, 23, 26, 17, 12, 30, 4]
PlotSampleID = [0, 1, 2,  5, 9, 4, 7, 12, 14]

max_relError = 0
ErrTh = 0.5
PlotRatio = True

W_predArray = np.load('encoded_test_GP.npy')  ## From Mickael
# W_varArray = np.load('Var_preds.npy')  ## From Mickael


if PlotRatio:


    Cl_Original = (normFactor*x_test)#[2:3]
    RealParaArray = y_test#[2:3]



    # Cl_Original = (normFactor*x_train)[0:10]
    # RealParaArray = y_train[0:10]



    for i in range(np.shape(RealParaArray)[0]):

        RealPara = RealParaArray[i]


        W_pred = np.array([W_predArray[i]])
        # x_decoded = decoder.predict(W_pred*ymax)# + meanFactor
        x_decoded = decoder.predict(W_pred)# + meanFactor


        plt.figure(94, figsize=(8,6))
        plt.title('Autoencoder+GP fit')
        # cl_ratio = 10**(normFactor*x_decoded[0])/10**(Cl_Original[i])

        cl_ratio = (normFactor*x_decoded[0])/(Cl_Original[i])


        relError = 100*((cl_ratio) - 1)

        plt.plot(ls, cl_ratio, alpha=.8, lw = 1.0)
        plt.ylim(0.95, 1.05)
        # plt.xscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$C_l^{GPAE}$/$C_l^{Original}$')
        plt.title(fileOut)
        # plt.legend()
        plt.tight_layout()



        if i in PlotSampleID:

            plt.figure(99, figsize=(8,6))
            plt.title('Autoencoder+GP fit')
            # plt.plot(ls, normFactor * x_test[::].T, 'gray', alpha=0.1)

            # plt.plot(ls, 10**(normFactor*x_decoded[0]), 'r--', alpha= 0.5, lw = 1, label = 'emulated')
            # plt.plot(ls, 10**(Cl_Original[i]), 'b--', alpha=0.5, lw = 1, label = 'original')

            plt.plot(ls, (normFactor*x_decoded[0]), 'r--', alpha= 0.8, lw = 1, label = 'emulated')
            plt.plot(ls, (Cl_Original[i]), 'b--', alpha=0.8, lw = 1, label = 'original')


            # plt.xscale('log')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$C_l$')
            plt.legend()
            # plt.tight_layout()

            plt.plot(ls[np.abs(relError) > ErrTh], normFactor*x_decoded[0][np.abs(relError) >
                                                                          ErrTh], 'gx',
                     alpha=0.7, label='bad eggs', markersize = '1')
            plt.title(fileOut)

            plt.savefig(PlotsDir + 'TestP'+str(num_para)+''+fileOut+'.png')
        #plt.show()
        print(i, 'ERR min max:', np.array([(relError).min(), (relError).max()]) )

        max_relError = np.max( [np.max(np.abs(relError)) , max_relError] )

    plt.figure(94, figsize=(8,6))
    plt.axhline(y=1, ls='-.', lw=1.5)
    plt.savefig(PlotsDir + 'RatioP'+str(num_para)+''+fileOut+'.png')

    #plt.show()
print(50*'-')
print('file:', fileOut)
# ------------------------------------------------------------------------------


plotLoss = True
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

print(50*'#')
print(fileOut)
print('train loss: ', train_loss[-1])
print('test loss: ', val_loss[-1])
print
print('max rel error:', max_relError)
print(50*'#')




# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


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
    plt.show()




# ------------------------------------------------------------------------------
