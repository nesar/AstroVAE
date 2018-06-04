import numpy as np
import matplotlib.pylab as plt
import corner
import emcee
import time
from keras.models import load_model
import params
import george
from george.kernels import Matern32Kernel


import pygtc


# import Cl_load
# import SetPub
# SetPub.set_pub()


#### parameters that define the MCMC

ndim = 5
nwalkers = 800  # 500
nrun_burn = 50  # 300
nrun = 400 # 700
fileID = 1


########## REAL DATA with ERRORS #############################
# Planck/SPT/WMAP data
# TE, EE, BB next

dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt']



### SPT TT
lID = np.array([0, 2, 0])
ClID = np.array([1, 3, 1])
emaxID = np.array([2, 4, 2])
eminID = np.array([2, 4, 2])

print(allfiles)


# for fileID in [realDataID]:
with open(dirIn + allfiles[fileID]) as f:
    lines = (line for line in f if not line.startswith('#'))
    allCl = np.loadtxt(lines, skiprows=1)

    l = allCl[:, lID[fileID]].astype(int)
    Cl = allCl[:, ClID[fileID]]
    emax = allCl[:, emaxID[fileID]]
    emin = allCl[:, eminID[fileID]]

    print(l.shape)




fileID = 1


dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt']



### SPT EE
lID = np.array([0, 5, 0])
ClID = np.array([1, 6, 1])
emaxID = np.array([2, 7, 2])
eminID = np.array([2, 7, 2])

print(allfiles)


# for fileID in [realDataID]:
with open(dirIn + allfiles[fileID]) as f:
    lines = (line for line in f if not line.startswith('#'))
    allCl = np.loadtxt(lines, skiprows=1)

    l2 = allCl[:, lID[fileID]].astype(int)
    Cl2 = allCl[:, ClID[fileID]]
    emax2 = allCl[:, emaxID[fileID]]
    emin2 = allCl[:, eminID[fileID]]

    print(l2.shape)




fileID = 1


dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt']



### SPT TE
lID = np.array([0, 8, 0])
ClID = np.array([1, 9, 1])
emaxID = np.array([2, 10, 2])
eminID = np.array([2, 10, 2])

print(allfiles)


# for fileID in [realDataID]:
with open(dirIn + allfiles[fileID]) as f:
    lines = (line for line in f if not line.startswith('#'))
    allCl = np.loadtxt(lines, skiprows=1)

    l3 = allCl[:, lID[fileID]].astype(int)
    Cl3 = allCl[:, ClID[fileID]]
    emax3 = allCl[:, emaxID[fileID]]
    emin3 = allCl[:, eminID[fileID]]

    print(l3.shape)



############## GP FITTING ################################################################################
##########################################################################################################



def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


###################### PARAMETERS ##############################

original_dim = params.original_dim  # 2549
latent_dim = params.latent_dim  # 10

# ClID = params.ClID
num_train = params.num_train  # 512
num_test = params.num_test  # 32
num_para = params.num_para  # 5

batch_size = params.batch_size  # 8
num_epochs = params.num_epochs  # 100
epsilon_mean = params.epsilon_mean  # 1.0
epsilon_std = params.epsilon_std  # 1.0
learning_rate = params.learning_rate  # 1e-3
decay_rate = params.decay_rate  # 0.0

noise_factor = params.noise_factor  # 0.00

######################## I/O ##################################

DataDir = params.DataDir
PlotsDir = params.PlotsDir
ModelDir = params.ModelDir

fileOut = params.fileOut

# ----------------------------- i/o ------------------------------------------




ls = np.loadtxt(DataDir + 'P' + str(num_para) + 'ls_' + str(num_train) + '.txt')[2:]


import GPy

# ----------------------------- TT ------------------------------------


ClID = ['TT', 'EE', 'BB', 'TE'][0]


normFactor = np.loadtxt(DataDir + 'normfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')
meanFactor = np.loadtxt(DataDir + 'meanfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')

print('-------normalization factor:', normFactor)
print('-------rescaling factor:', meanFactor)


GPmodelOutfile = DataDir + 'GPy_model' + str(latent_dim) + ClID + fileOut
m1 = GPy.models.GPRegression.load_model(GPmodelOutfile + '.zip')


decoderFile = ModelDir + 'DecoderP' + str(num_para) + ClID + '_' + fileOut + '.hdf5'
decoder = load_model(decoderFile)



def GPyfit(para_array):

    test_pts = para_array.reshape(num_para, -1).T

    # -------------- Predict latent space ----------------------------------------

    m1p = m1.predict(test_pts)  # [0] is the mean and [1] the predictive
    W_pred = m1p[0]

    # -------------- Decode from latent space --------------------------------------

    x_decoded = decoder.predict(W_pred.reshape(latent_dim, -1).T )

    return (normFactor * x_decoded[0]) + meanFactor



# ----------------------------- EE ------------------------------------

ClID = ['TT', 'EE', 'BB', 'TE'][1]


normFactor2 = np.loadtxt(DataDir + 'normfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')
meanFactor2 = np.loadtxt(DataDir + 'meanfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')

print('-------normalization factor:', normFactor2)
print('-------rescaling factor:', meanFactor2)


GPmodelOutfile = DataDir + 'GPy_model' + str(latent_dim) + ClID + fileOut
m2 = GPy.models.GPRegression.load_model(GPmodelOutfile + '.zip')


decoderFile = ModelDir + 'DecoderP' + str(num_para) + ClID + '_' + fileOut + '.hdf5'
decoder2 = load_model(decoderFile)


def GPyfit2(para_array):

    test_pts = para_array.reshape(num_para, -1).T


    m2p = m2.predict(test_pts)  # [0] is the mean and [1] the predictive
    W_pred = m2p[0]

    # -------------- Decode from latent space --------------------------------------

    x_decoded = decoder2.predict(W_pred.reshape(latent_dim, -1).T )

    return (normFactor2 * x_decoded[0]) + meanFactor2



# ----------------------------- TE ------------------------------------

ClID = ['TT', 'EE', 'BB', 'TE'][3]


normFactor3 = np.loadtxt(DataDir + 'normfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')
meanFactor3 = np.loadtxt(DataDir + 'meanfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')

print('-------normalization factor:', normFactor2)
print('-------rescaling factor:', meanFactor2)


GPmodelOutfile = DataDir + 'GPy_model' + str(latent_dim) + ClID + fileOut
m3 = GPy.models.GPRegression.load_model(GPmodelOutfile + '.zip')


decoderFile = ModelDir + 'DecoderP' + str(num_para) + ClID + '_' + fileOut + '.hdf5'
decoder3 = load_model(decoderFile)


def GPyfit3(para_array):

    test_pts = para_array.reshape(num_para, -1).T


    m3p = m3.predict(test_pts)  # [0] is the mean and [1] the predictive
    W_pred = m3p[0]

    # -------------- Decode from latent space --------------------------------------

    x_decoded = decoder3.predict(W_pred.reshape(latent_dim, -1).T )

    return (normFactor3 * x_decoded[0]) + meanFactor3

######### ---------------------------------------------------------------------------


########################################################################################################################
########################################################################################################################

#### Cosmological Parameters ########################################

#### Order of parameters: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s']
#        [label, true, min, max]



## Make sure the changes are made in log prior definition too. Variable: new_params


param1 = ["$\Omega_c h^2$", 0.1197, 0.10, 0.14] #
param2 = ["$\Omega_b h^2$", 0.02222, 0.0205, 0.0235]
param3 = ["$\sigma_8$", 0.829, 0.7, 0.9]
param4 = ["$h$", 0.6731, 0.55, 0.85]
param5 = ["$n_s$", 0.9655, 0.85, 1.05]



#################### CHAIN INITIALIZATION ##########################

## 2 options

Uniform_init = True
if Uniform_init:
# Choice 1: chain uniformly distributed in the range of the parameters
    pos_min = np.array([param1[2], param2[2], param3[2], param4[2], param5[2]])
    pos_max = np.array([param1[3], param2[3], param3[3], param4[3], param5[3]])
    psize = pos_max - pos_min
    pos0 = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]



True_init = False
if True_init:
# Choice 2: chain is initialized in a tight ball around the expected values
    pos0 = [[param1[1]*1.2, param2[1]*0.8, param3[1]*0.9, param4[1]*1.1, param5[1]*1.2] +
           1e-3*np.random.randn(ndim) for i in range(nwalkers)]

#
# MaxLikelihood_init = False
# if MaxLikelihood_init:
# # Choice 2b: Find expected values from max likelihood and use that for chain initialization
# # Requires likehood function below to run first
#
#     import scipy.optimize as op
#     nll = lambda *args: -lnlike(*args)
#     result = op.minimize(nll, [param1[1], param2[1], param3[1], param4[1], param5[1]], args=(x, y, yerr))
#     p1_ml, p2_ml, p3_ml, p4_ml, p5_ml = result["x"]
#     print result['x']
#
#
#     pos0 = [result['x']+1.e-4*np.random.randn(ndim) for i in range(nwalkers)]



# Visualize the initialization

PriorPlot = False

if PriorPlot:

    fig = corner.corner(pos0, labels=[param1[0], param2[0], param3[0], param4[0], param5[0]],
                        range=[[param1[2],param1[3]], [param2[2], param2[3]], [param3[2],param3[3]],
                        [param4[2],param4[3]], [param5[2],param5[3]]],
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]])
    fig.set_size_inches(10, 10)




######### MCMC #######################


x = l[l < ls.max()]
y = Cl[l < ls.max()]
yerr = emax[l < ls.max()]



x2 = l2[l2 < ls.max()]
y2 = Cl2[l2 < ls.max()]
yerr2 = emax2[l2 < ls.max()]



x3 = l3[l3 < ls.max()]
y3 = Cl3[l3 < ls.max()]
yerr3 = emax3[l3 < ls.max()]


## Sample implementation :
# http://eso-python.github.io/ESOPythonTutorials/ESOPythonDemoDay8_MCMC_with_emcee.html
# https://users.obs.carnegiescience.edu/cburns/ipynbs/Emcee.html

def lnprior(theta):
    p1, p2, p3, p4, p5 = theta
    # if 0.12 < p1 < 0.155 and 0.7 < p2 < 0.9:
    if param1[2] < p1 < param1[3] and param2[2] < p2 < param2[3] and param3[2] < p3 < param3[3] \
            and param4[2] < p4 < param4[3] and param5[2] < p5 < param5[3]:
        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr):
    p1, p2, p3, p4, p5 = theta
    # new_params = np.array([p1, 0.0225, p2 , 0.74, 0.9])

    new_params = np.array([p1, p2, p3, p4, p5])
    # model = GPfit(computedGP, new_params)#  Using George -- with model training

    model = GPyfit(new_params)# Using GPy -- using trained model


    mask = np.in1d(ls, x)
    model_mask = model[mask]

    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model_mask) / yerr) ** 2.))





def lnlike2(theta, x, y, yerr):
    p1, p2, p3, p4, p5 = theta

    new_params = np.array([p1, p2, p3, p4, p5])

    model = GPyfit2(new_params)# Using GPy -- using trained model


    mask = np.in1d(ls, x)
    model_mask = model[mask]

    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model_mask) / yerr) ** 2.))



def lnlike3(theta, x, y, yerr):
    p1, p2, p3, p4, p5 = theta

    new_params = np.array([p1, p2, p3, p4, p5])

    model = GPyfit3(new_params)# Using GPy -- using trained model


    mask = np.in1d(ls, x)
    model_mask = model[mask]

    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model_mask) / yerr) ** 2.))



def lnprob(theta, x, y, yerr, x2, y2, yerr2, x3, y3, yerr3):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr) + lnlike2(theta, x2, y2, yerr2) + lnlike3(theta, x3,
                                                                                      y3, yerr3)



########## ------------------------------------------------------------------ #########

# Let us setup the emcee Ensemble Sampler
# It is very simple: just one, self-explanatory line

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, x2, y2, yerr2, x3, y3,
                                                              yerr3))

###### BURIN-IN #################

time0 = time.time()
# burnin phase
pos, prob, state = sampler.run_mcmc(pos0, nrun_burn)
sampler.reset()
time1 = time.time()
print('burn-in time:', time1 - time0)

###### MCMC ##################
time0 = time.time()
# perform MCMC
pos, prob, state = sampler.run_mcmc(pos, nrun)
time1 = time.time()
print('mcmc time:', time1 - time0)

samples = sampler.flatchain
samples.shape


###########################################################################
samples_plot = sampler.chain[:, :, :].reshape((-1, ndim))



np.savetxt(DataDir + 'Sampler_mcmcSPT_ndim' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
    nrun)  + ClID + '_'  + fileOut + allfiles[fileID][:-4] +'.txt', sampler.chain[:, :, :].reshape((-1, ndim)))

####### FINAL PARAMETER ESTIMATES #######################################


samples_plot  = np.loadtxt(DataDir + 'Sampler_mcmcSPT_ndim' + str(ndim) + '_nwalk' + str(nwalkers) +
                         '_run' + str(nrun)  + ClID + '_' + fileOut + allfiles[fileID][:-4] +'.txt')

# samples = np.exp(samples)
p1_mcmc, p2_mcmc, p3_mcmc, p4_mcmc, p5_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                       zip(*np.percentile(samples_plot, [16, 50, 84], axis=0)))
print('mcmc results:', p1_mcmc[0], p2_mcmc[0], p3_mcmc[0], p4_mcmc[0], p5_mcmc[0])

####### CORNER PLOT ESTIMATES #######################################

CornerPlot = False
if CornerPlot:
    #
    fig = corner.corner(samples_plot, labels=[param1[0], param2[0], param3[0], param4[0], param5[0]],
                        range=[[param1[2],param1[3]], [param2[2], param2[3]], [param3[2],param3[3]],
                        [param4[2],param4[3]], [param5[2],param5[3]]],
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]],
                        show_titles=True,  title_args={"fontsize": 10})


    fig = pygtc.plotGTC(samples_plot, paramNames=[param1[0], param2[0], param3[0], param4[0], param5[0]],
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]],
                        figureSize='MNRAS_page')#, plotDensity = True, filledPlots = False,\smoothingKernel = 0, nContourLevels=3)


    fig.savefig(PlotsDir + 'SPT_pygtc_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun)  + ClID + '_' + fileOut + allfiles[fileID][:-4] +'.pdf')



CornerCompare = False

if CornerCompare:
    ndim = 5
    nwalkers = 600  # 500
    nrun_burn = 50  # 300
    nrun = 300  # 700
    ClID = 'TT'

    samples_plotSPT = np.loadtxt(
        DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) +
        '_run' + str(nrun) + ClID + '_' + fileOut + allfiles[fileID][:-4] + '.txt')

    fig = pygtc.plotGTC(chains=[samples_plot, samples_plotSPT], paramNames=[param1[0], param2[0], param3[0], param4[0], param5[0]],
                        colorsOrder=('reds', 'blues'), chainLabels=["SPT TT+EE+TE", "SPT TT"],
                        figureSize='MNRAS_page')  # , plotDensity = True, filledPlots = False,\smoothingKernel = 0, nContourLevels=3)


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

ConvergePlot = False
if ConvergePlot:

    fig = plt.figure(13214)
    plt.xlabel('steps')
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5)

    ax1.plot(np.arange(nrun), sampler.chain[:, :, 0].T, lw = 0.2, alpha = 0.9)
    ax1.text(0.9, 0.9, param1[0], horizontalalignment='center', verticalalignment='center',
             transform = ax1.transAxes, fontsize = 20)
    ax2.plot(np.arange(nrun), sampler.chain[:, :, 1].T, lw = 0.2, alpha = 0.9)
    ax2.text(0.9, 0.9, param2[0], horizontalalignment='center', verticalalignment='center',
             transform = ax2.transAxes, fontsize = 20)
    ax3.plot(np.arange(nrun), sampler.chain[:, :, 2].T, lw = 0.2, alpha = 0.9)
    ax3.text(0.9, 0.9, param3[0], horizontalalignment='center', verticalalignment='center',
             transform = ax3.transAxes, fontsize = 20)
    ax4.plot(np.arange(nrun), sampler.chain[:, :, 3].T, lw = 0.2, alpha = 0.9)
    ax4.text(0.9, 0.9, param4[0], horizontalalignment='center', verticalalignment='center',
             transform = ax4.transAxes, fontsize = 20)
    ax5.plot(np.arange(nrun), sampler.chain[:, :, 4].T, lw = 0.2, alpha = 0.9)
    ax5.text(0.9, 0.9, param5[0], horizontalalignment='center', verticalalignment='center',
             transform = ax5.transAxes, fontsize = 20)
    plt.show()

    fig.savefig(PlotsDir + 'convergence_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun)  + ClID + '_'  + fileOut + allfiles[fileID][:-4] +'.pdf')





### Comapring with real values


y_mcmc = np.array([ p1_mcmc[0], p2_mcmc[0], p3_mcmc[0], p4_mcmc[0], p5_mcmc[0]] )
x_decodedGPy = GPyfit(y_mcmc)
x_decodedGPy2 = GPyfit2(y_mcmc)
x_decodedGPy3 = GPyfit3(y_mcmc)

yPLANCK2015 = np.array([param1[1], param2[1], param3[1], param4[1], param5[1]])
x_decodedGPy2015 = GPyfit(yPLANCK2015)
x_decodedGPy2015_2 = GPyfit2(yPLANCK2015)
x_decodedGPy2015_3 = GPyfit3(yPLANCK2015)





plt.figure(32534, figsize=(6, 9))
from matplotlib import gridspec

gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,1])
gs.update(hspace=0.02, left=0.2, bottom = 0.15)  # set the spacing between axes.


ax0 = plt.subplot(gs[0])
ax0.errorbar(l, Cl, yerr = [emin, emax], fmt='o', mec = 'k', ms=2, ecolor = 'k', elinewidth=0.8,
             alpha = 0.8, label = 'SPTpol data')
ax0.plot(ls, x_decodedGPy, 'r--', alpha = 0.7, label = 'Best estimate')
ax0.plot(ls, x_decodedGPy2015, 'b--', alpha = 0.7, label = 'PLANCK estimate')


ax0.set_ylabel(r'$l(l+1)C_l^{TT}/2\pi [\mu K^2]$')
ax0.set_xlabel(r'$l$')


ax1 = plt.subplot(gs[1])
ax1.errorbar(l2, Cl2, yerr = [emin2, emax2], fmt='o', mec = 'k', ms=2, ecolor = 'k', elinewidth=0.8,
             alpha = 0.99, label = 'SPTpol data')
ax1.plot(ls, x_decodedGPy2, 'r--', alpha = 0.7, label = 'Best estimate')
ax1.plot(ls, x_decodedGPy2015_2, 'b--', alpha = 0.7, label = 'PLANCK estimate')


ax1.set_ylabel(r'$l(l+1)C_l^{EE}/2\pi [\mu K^2]$')
ax1.set_xlabel(r'$l$')


ax2 = plt.subplot(gs[2])
ax2.errorbar(l3, Cl3, yerr = [emin3, emax3], fmt='o', mec = 'k', ms=2, ecolor = 'k', elinewidth=0.8,
             alpha = 0.99, label = 'SPTpol data')
ax2.plot(ls, x_decodedGPy3, 'r--', alpha = 0.7, label = 'Best estimate')
ax2.plot(ls, x_decodedGPy2015_3, 'b--', alpha = 0.7, label = 'PLANCK estimate')



ax2.set_ylabel(r'$l(l+1)C_l^{TE}/2\pi [\mu K^2]$')
ax2.set_xlabel(r'$l$')

ax0.set_xlim(0, 2500)
ax1.set_xlim(0, 2500)
ax2.set_xlim(0, 2500)

ax0.legend(title = 'TT')
ax1.legend(title = 'EE')
ax2.legend(title = 'TE')

plt.savefig(PlotsDir + 'SPT_TT_EE_TE_BestFit.pdf')