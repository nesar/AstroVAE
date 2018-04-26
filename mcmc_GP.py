import numpy as np
import matplotlib.pylab as plt


# Choose the "true" parameters.

# cosmological parameters here


########## REAL DATA with ERRORS #############################
# Generate some synthetic data from the model.


# N = 50
# x = np.sort(10*np.random.rand(N))
# yerr = 0.1+0.5*np.random.rand(N)
# y = m_true*x+b_true
# y += np.abs(f_true*y) * np.random.randn(N)
# y += yerr * np.random.randn(N)


dirIn = '../../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt']

# lID = np.array([1, 0, 2, 0])
# ClID = np.array([1, 1, 3, 1])
# emaxID = np.array([1, 2, 4, 2])
# eminID = np.array([1, 2, 4, 2])


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


XY = np.array(np.array([X1a, X2a, X3a, X4a, X5a])[:, :, 0])[:, np.newaxis]


# # ------------------------------------------------------------------------------
encoded_xtrain = np.loadtxt(DataDir + 'encoded_xtrainP'+str(num_para)+ClID+'_'+ fileOut +'.txt').T
encoded_xtest_original = np.loadtxt(DataDir+'encoded_xtestP'+str(num_para)+ClID+'_'+ fileOut +'.txt')

# ------------------------------------------------------------------------------

def GPcompute(XY, latent_dim):
    gp = {}
    for j in range(latent_dim):
        gp["fit{0}".format(j)] = george.GP(kernel)
        gp["fit{0}".format(j)].compute(XY[:, 0, :].T)
    return gp


def GPfit(computedGP, y_params):
    RealPara = y_params


    RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
    RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
    RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2])
    RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
    RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])

    test_pts = RealPara[:num_para].reshape(num_para, -1).T

    # ------------------------------------------------------------------------------

    W_pred = np.array([np.zeros(shape=latent_dim)])
    W_pred_var = np.array([np.zeros(shape=latent_dim)])

    gp = computedGP
    for j in range(latent_dim):
        W_pred[:, j], W_pred_var[:, j] = gp["fit{0}".format(j)].predict(encoded_xtrain[j], test_pts)

    x_decoded = decoder.predict(W_pred)

    return (normFactor* x_decoded[0])+meanFactor


computedGP = GPcompute(XY, latent_dim)

x_decoded = GPfit(computedGP, y_test[10])

#
# plt.figure(1423)
# plt.plot(l, x_decoded[28:2507])
# plt.plot(x, y, alpha = 0.4)
# plt.show()


########################################################################################################################
########################################################################################################################


## **** checl camb generate pars[2]

### recheck the values!!

### Try using just 2 parameters --- m, b  -- 0(omega_m), 2(sigma_8)
### x should cut the x_decoded part
### GPfit(computedGP, y_test[0]) should be in lnlike -- in model

m_true = 0.128  # omch2
b_true = 0.0225  #ombh2
f_true = 0.534  #dunno what this should do

######### MCMC #######################

# OmegaM = np.linspace(0.12, 0.155, totalFiles)
# Omegab = np.linspace(0.0215, 0.0235, totalFiles)
# sigma8 = np.linspace(0.7, 0.9, totalFiles)
# h = np.linspace(0.55, 0.85, totalFiles)
# ns = np.linspace(0.85, 1.05, totalFiles)


# def lnlike(theta, x, y, yerr):
#     m, b, lnf = theta
#     model = m * x + b
#     inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
#     return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# log likelihood function -- to be replaced by the emulator as a function of cosmological parameters

def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    new_params = np.array([m, b, 0.8 , 0.74, 0.9])
    model = GPfit(computedGP, new_params)[28:2507]
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

import scipy.optimize as op
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
# m_ml, b_ml, lnf_ml = result["x"]   ## max likelihood


#############################################################
# Use flat prior ?
# def lnprior(theta):
#     m, b, lnf = theta
#     if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
#         return 0.0
#     return -np.inf


def lnprior(theta):
    m, b, lnf = theta
    if 0.12 < m < 0.155 and 0.0215 < b < 0.0235 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf


# Dunno what to do here  -- just a combination of likelihood and prior?
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)



ndim, nwalkers = 3, 500  # 3,100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

nrun = 500

import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

# sampler.run_mcmc(pos, 500)
sampler.run_mcmc(pos, nrun)

np.savetxt(DataDir + 'Sampler_mcmc_ndim' +str(ndim) + '_nwalk' + str(nwalkers) + '_run' +  str(nrun) + fileOut +'.txt',  sampler.chain[:,:,:].reshape((-1, ndim)) )


samples = sampler.chain[:, 50:, :].reshape((-1, ndim))



####### CORNER PLOT ESTIMATES #######################################


import pygtc
import corner
# fig = corner.corner(samples, labels=["$\Omega_c h^2$", "$\sigma_8$", "$\ln\,f$"],
#                       truths=[m_true, b_true, np.log(f_true)])



samples_plot = sampler.chain[:, 50:, 0:2].reshape((-1, ndim-1))


fig = corner.corner(samples_plot, labels=["$\Omega_c h^2$", "$\Omega_b h^2$"],
                      truths=[m_true, b_true])
fig.savefig('corner_'+fileOut+'.png')


fig = pygtc.plotGTC(samples_plot, paramNames=["$\Omega_c h^2$", "$\Omega_b h^2$"],
                      truths=[m_true, b_true])
fig.savefig('pygtc_'+fileOut+'.png')

####### FINAL PARAMETER ESTIMATES #######################################

plt.figure(1432)

xl = np.array([0, 10])
for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
    plt.plot(xl, m*xl+b, color="k", alpha=0.1)
plt.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
plt.errorbar(x, y, yerr=yerr, fmt=".k", alpha=0.1)


####### FINAL PARAMETER ESTIMATES #######################################

samples[:, 2] = np.exp(samples[:, 2])
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
#####################################################################
