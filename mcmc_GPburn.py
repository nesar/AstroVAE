import numpy as np
import matplotlib.pylab as plt

########## REAL DATA with ERRORS #############################
# Generate some synthetic data from the model.


dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt']


lID = np.array([0, 2, 0])
ClID = np.array([1, 3, 1])
emaxID = np.array([2, 4, 2])
eminID = np.array([2, 4, 2])

print(allfiles)

import numpy as np

for fileID in [2]:
    with open(dirIn + allfiles[fileID]) as f:
        lines = (line for line in f if not line.startswith('#'))
        allCl = np.loadtxt(lines, skiprows=1)

    l = allCl[:, lID[fileID]]
    Cl = allCl[:, ClID[fileID]]
    emax = allCl[:, emaxID[fileID]]
    emin = allCl[:, eminID[fileID]]

    print(l.shape)

x = l
y = Cl
yerr = emax


############## GP FITTING ################################################################################
##########################################################################################################


from keras.models import load_model

import params
#import Cl_load
#import SetPub
#SetPub.set_pub()



def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)



###################### PARAMETERS ##############################


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


# ----------------------------- i/o ------------------------------------------


ClID = ['TT', 'EE', 'BB', 'TE'][0]

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
from george.kernels import Matern32Kernel

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


rescaledTrainParams = np.array(np.array([X1a, X2a, X3a, X4a, X5a])[:, :, 0])[:, np.newaxis]


# # ------------------------------------------------------------------------------
encoded_xtrain = np.loadtxt(DataDir + 'encoded_xtrainP'+str(num_para)+ClID+'_'+ fileOut +'.txt').T
encoded_xtest_original = np.loadtxt(DataDir+'encoded_xtestP'+str(num_para)+ClID+'_'+ fileOut +'.txt')

# ------------------------------------------------------------------------------

def GPcompute(rescaledTrainParams, latent_dim):
    gp = {}
    for j in range(latent_dim):
        gp["fit{0}".format(j)] = george.GP(kernel)
        gp["fit{0}".format(j)].compute(rescaledTrainParams[:, 0, :].T)
    return gp


def GPfit(computedGP, para_array):
    
    para_array[0] = rescale01(np.min(X1), np.max(X1), para_array[0])
    para_array[1] = rescale01(np.min(X2), np.max(X2), para_array[1])
    para_array[2] = rescale01(np.min(X3), np.max(X3), para_array[2])
    para_array[3] = rescale01(np.min(X4), np.max(X4), para_array[3])
    para_array[4] = rescale01(np.min(X5), np.max(X5), para_array[4])

    test_pts = para_array[:num_para].reshape(num_para, -1).T

    # -------------- Predict latent space ----------------------------------------

    W_pred = np.array([np.zeros(shape=latent_dim)])
    W_pred_var = np.array([np.zeros(shape=latent_dim)])

    for j in range(latent_dim):
        W_pred[:, j], W_pred_var[:, j] = computedGP["fit{0}".format(j)].predict(encoded_xtrain[j], test_pts)


    # -------------- Decode from latent space --------------------------------------

    x_decoded = decoder.predict(W_pred)

    return (normFactor* x_decoded[0])+meanFactor


computedGP = GPcompute(rescaledTrainParams, latent_dim)

x_decoded = GPfit(computedGP, y_test[10])

#
# plt.figure(1423)
# plt.plot(l, x_decoded[28:2507])
# plt.plot(x, y, alpha = 0.4)
# plt.show()


########################################################################################################################
########################################################################################################################



#### parameters that define the MCMC
ndim = 2
nwalkers = 50 #500
nrun_burn = 100 #300
nrun = 700 #700


#### Cosmological Parameters ########################################

#### Order of parameters: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s']
#        [label, true, min, max]

param1 = ["$\Omega_c h^2$", 0.121, 0.12, 0.155]
# param2 = ["$\sigma_8$", 0.81, 0.7, 0.9]
param2 = ["$h$", 0.67, 0.55, 0.85]

## Make sure the changes are made in log prior definition too. Variable: new_params




###################################################################


# Initialize the chain
# Choice 1: chain uniformly distributed in the range of the parameters
pos_min = np.array([param1[2], param2[2]])
pos_max = np.array([param1[3], param2[3]])
psize = pos_max - pos_min
pos = [pos_min + psize*np.random.rand(ndim) for i in range(nwalkers)]

# Visualize the initialization
import corner
fig = corner.corner(pos, labels=[param1[0], param2[0]], range=[[param1[2], param1[3]], [param2[2],
                                                                                  param2[3]]],
                      truths=[param1[1], param2[1]])
fig.set_size_inches(10,10)

######### MCMC #######################

# OmegaM = np.linspace(0.12, 0.155, totalFiles)
# Omegab = np.linspace(0.0215, 0.0235, totalFiles)
# sigma8 = np.linspace(0.7, 0.9, totalFiles)
# h = np.linspace(0.55, 0.85, totalFiles)
# ns = np.linspace(0.85, 1.05, totalFiles)


def lnprior(theta):
    p1, p2= theta
    # if 0.12 < p1 < 0.155 and 0.7 < p2 < 0.9:
    if param1[2] < p1 < param1[3] and param2[2] < p2 < param2[3]:

        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr):
    p1, p2 = theta
    # new_params = np.array([p1, 0.0225, p2 , 0.74, 0.9])

    new_params = np.array([p1, 0.0225, 0.1201 , p2, 0.9])
    model = GPfit(computedGP, new_params)[28:2507]

    return -0.5*(np.sum( (  (y-model)/yerr  )**2. ))


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)



# Let us setup the emcee Ensemble Sampler
# It is very simple: just one, self-explanatory line
import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))






###### BURIN-IN #################
import time
time0 = time.time()
# burnin phase
pos, prob, state  = sampler.run_mcmc(pos, nrun_burn)
sampler.reset()
time1=time.time()
print('burn-in time:', time1-time0)



###### MCMC ##################
time0 = time.time()
# perform MCMC
pos, prob, state  = sampler.run_mcmc(pos, nrun)
time1=time.time()
print('mcmc time:', time1-time0)

samples = sampler.flatchain
samples.shape







samples_plot = sampler.chain[:, :].reshape((-1, ndim))


fig = corner.corner(samples_plot, labels=[param1[0], param2[0]],
                      truths=[param1[1], param2[1]])
fig.savefig('corner_'+fileOut+'.png')



np.savetxt(DataDir + 'Sampler_mcmc_ndim' +str(ndim) + '_nwalk' + str(nwalkers) + '_run' +  str(nrun) + fileOut +'.txt',  sampler.chain[:,:,:].reshape((-1, ndim)) )


####### FINAL PARAMETER ESTIMATES #######################################

# samples = np.exp(samples)
p1_mcmc, p2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50, 84], axis=0)))
print('mcmc results:', p1_mcmc[0], p2_mcmc[0])


####### CORNER PLOT ESTIMATES #######################################


import corner

fig = corner.corner(samples_plot, labels=[param1[0], param2[0]],
                      truths=[param1[1], param2[1]])
fig.savefig('corner_'+fileOut+'.png')



import pygtc
fig = pygtc.plotGTC(samples_plot, paramNames=[param1[0], param2[0]], truths=[ param1[1],
                                                                             param2[1]] ,
                    figureSize='MNRAS_page' )
fig.savefig('pygtc_'+fileOut+'.png')



####### FINAL PARAMETER ESTIMATES #######################################
#
# plt.figure(1432)
#
# xl = np.array([0, 10])
# for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
#     plt.plot(xl, m*xl+b, color="k", alpha=0.1)
# plt.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
# plt.errorbar(x, y, yerr=yerr, fmt=".k", alpha=0.1)



####### SAMPLER CONVERGENCE #######################################


fig = plt.figure(13214)
ax1 = fig.add_subplot(1, 1, 1)
ax2 = fig.add_subplot(2, 1, 1)
# ax3 = fig.add_subplot(3, 1, 3)

ax1.plot(np.arange(nrun-10), sampler.chain[:,10:, 0].T)
ax2.plot(np.arange(nrun-10), sampler.chain[:,10:, 1].T)
# ax3.plot(np.arange(nrun), sampler.chain[:,:, 2].T)

plt.show()
