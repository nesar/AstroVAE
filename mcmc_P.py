# import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import matplotlib.pylab as plt
import emcee
import time
import params
import pygtc
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


#### parameters that define the MCMC

ndim = 4
nwalkers = 20 #200 #600  # 500
nrun_burn = 10 # 50 # 50  # 300
nrun = 30 # 300  # 700
fileID = 1


############################# PARAMETERS ##############################


DataDir = params.DataDir
PlotsDir = params.PlotsDir
ModelDir = params.ModelDir

fileOut = params.fileOut

# ----------------------------- i/o ------------------------------------------


import numpy as np
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from astropy.io import fits as pf
import astropy.table

RcppCNPy = importr('RcppCNPy')
# RcppCNPy.chooseCRANmirror(ind=1) # select the first mirror in the list

########

################################# I/O #################################
fitsfileIn =  "../P_data/2ndpass_vals_for_test.fits"

Allfits =pf.open(fitsfileIn)
AllData =astropy.table.Table(Allfits[1].data)

parameter_array = np.array([AllData['RHO'], AllData['SIGMA_LAMBDA'], AllData['TAU'], 
                         AllData['SSPT']]).T

nr, nc = parameter_array.shape
u_train = ro.r.matrix(parameter_array, nrow=nr, ncol=nc)

ro.r.assign("u_train2", u_train)
r('dim(u_train2)')

pvec = (AllData['PVEC'])#.newbyteorder('S')
# print(  np.unique( np.argwhere( np.isnan(pvec) )[:,0]) )

np.savetxt('pvec.txt', pvec)
pvec = np.loadtxt('pvec.txt')

nr, nc = pvec.shape
y_train = ro.r.matrix(pvec, nrow=nr, ncol=nc)

ro.r.assign("y_train2", y_train)
r('dim(y_train2)')


########################### PCA ###################################

Dicekriging = importr('DiceKriging')

r('require(foreach)')

r('svd(y_train2)')

r('nrankmax <- 64')   ## Number of components

r('svd_decomp2 <- svd(y_train2)')
r('svd_weights2 <- svd_decomp2$u[, 1:nrankmax] %*% diag(svd_decomp2$d[1:nrankmax])')

######################## GP FITTING ################################
####################################################################
## Build GP models
GPareto = importr('GPareto')

r('''if(file.exists("R_GP_models_2.RData")){
        load("R_GP_models_2.RData")
    }else{
        models_svd2 <- list()
        for (i in 1: nrankmax){
            mod_s <- km(~., design = u_train2, response = svd_weights2[, i])
            models_svd2 <- c(models_svd2, list(mod_s))
        }
        save(models_svd2, file = "R_GP_models.RData")
        
     }''')

r('''''')

######################### INFERENCE ##################################


def GP_fit(para_array):

    test_pts = para_array
    test_pts = np.expand_dims(test_pts, axis=0)


    B = test_pts

    nr,nc = B.shape
    Br = ro.r.matrix(B, nrow=nr, ncol=nc)

    ro.r.assign("Br", Br)


    r('wtestsvd2 <- predict_kms(models_svd2, newdata = Br , type = "UK")')
    r('reconst_s2 <- t(wtestsvd2$mean) %*% t(svd_decomp2$v[,1:nrankmax])')


    y_recon = np.array(r('reconst_s2'))

    return y_recon[0]





plt.figure(999, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom = 0.15)  # set the spacing between axes.
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$P(x)$')
# # ax0.set_title( r'$\text{' +fileOut + '}$')

ax1.axhline(y=1, ls='dotted')
# ax1.axhline(y=-1e-6, ls='dashed')
# ax1.axhline(y=1e-6, ls='dashed')

ax1.set_xlabel(r'$x$')

ax1.set_ylabel(r'emu/real')
ax1.set_ylim(-1e-5, 1e-5)



##################################### TESTING ##################################



for x_id in [3, 23 , 43, 64, 93, 109, 121]:

    x_decodedGPy = GP_fit(parameter_array[x_id])   ## input parameters
    x_test = pvec[x_id]

    # plt.figure(1423)

    # plt.plot(x_decoded, 'k--', alpha = 0.4, label = 'George')
    ax0.plot(x_decodedGPy, alpha = 1.0 , ls = '--', label = 'emu')
    ax0.plot(x_test, alpha = 0.9 , label = 'real')
    plt.legend()

    ax1.plot(x_decodedGPy[1:]/x_test[1:] - 1)




plt.show()


#
# import sys
# sys.exit()


########################################################################################################################
########################################################################################################################

#### Cosmological Parameters ########################################

# OmegaM = np.linspace(0.12, 0.155, totalFiles)
# Omegab = np.linspace(0.0215, 0.0235, totalFiles)
# sigma8 = np.linspace(0.7, 0.9, totalFiles)
# h = np.linspace(0.55, 0.85, totalFiles)
# ns = np.linspace(0.85, 1.05, totalFiles)

#### Order of parameters: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s']
#        [label, true, min, max]

param1 = ["$\Omega_c h^2$", 0.1188, 0.12, 0.155] # Actual 0.119
param2 = ["$\Omega_b h^2$", 0.02230, 0.0215, 0.0235]
param3 = ["$\sigma_8$", 0.8159, 0.7, 0.9]
param4 = ["$h$", 0.6774, 0.55, 0.85]
param5 = ["$n_s$", 0.9667, 0.85, 1.05]

## Make sure the changes are made in log prior definition too. Variable: new_params


#
# OmegaM = np.linspace(0.10, 0.140, totalFiles)
# Omegab = np.linspace(0.0205, 0.0235, totalFiles)
# sigma8 = np.linspace(0.7, 0.9, totalFiles)
# h = np.linspace(0.55, 0.85, totalFiles)
# ns = np.linspace(0.85, 1.05, totalFiles)

param1 = ["$\Omega_c h^2$", 0.1188, 0.10, 0.14] # Actual 0.119
param2 = ["$\Omega_b h^2$", 0.02230, 0.0205, 0.0235]
param3 = ["$\sigma_8$", 0.8159, 0.7, 0.9]
param4 = ["$h$", 0.6774, 0.55, 0.85]
param5 = ["$n_s$", 0.9667, 0.85, 1.05]



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


MaxLikelihood_init = False
if MaxLikelihood_init:
# Choice 2b: Find expected values from max likelihood and use that for chain initialization
# Requires likehood function below to run first

    import scipy.optimize as op
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, [param1[1], param2[1], param3[1], param4[1], param5[1]], args=(x, y, yerr))
    p1_ml, p2_ml, p3_ml, p4_ml, p5_ml = result["x"]
    print result['x']


    pos0 = [result['x']+1.e-4*np.random.randn(ndim) for i in range(nwalkers)]



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

    model = GP_fit(new_params)# Using GPy -- using trained model


    mask = np.in1d(ls, x)
    model_mask = model[mask]

    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model_mask) / yerr) ** 2.))


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


# Let us setup the emcee Ensemble Sampler
# It is very simple: just one, self-explanatory line

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

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



np.savetxt(DataDir + 'SamplerPCA_mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
    nrun)  + ClID + '_'  + fileOut + allfiles[fileID][:-4] +'.txt', sampler.chain[:, :, :].reshape((-1, ndim)))

####### FINAL PARAMETER ESTIMATES #######################################


samples_plot  = np.loadtxt(DataDir + 'SamplerPCA_mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) +
                         '_run' + str(nrun)  + ClID + '_' + fileOut + allfiles[fileID][:-4] +'.txt')

# samples = np.exp(samples)
p1_mcmc, p2_mcmc, p3_mcmc, p4_mcmc, p5_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                       zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print('mcmc results:', p1_mcmc[0], p2_mcmc[0], p3_mcmc[0], p4_mcmc[0], p5_mcmc[0])

####### CORNER PLOT ESTIMATES #######################################

CornerPlot = False
if CornerPlot:

    fig = corner.corner(samples_plot, labels=[param1[0], param2[0], param3[0], param4[0], param5[0]],
                        range=[[param1[2],param1[3]], [param2[2], param2[3]], [param3[2],param3[3]],
                        [param4[2],param4[3]], [param5[2],param5[3]]],
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]],
                        show_titles=True,  title_args={"fontsize": 10})


    fig.savefig(PlotsDir +'cornerPCA_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun) + ClID + '_'  + fileOut +allfiles[fileID][:-4] + '.pdf')


    fig = pygtc.plotGTC(samples_plot, paramNames=[param1[0], param2[0], param3[0], param4[0], param5[0]],
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]],
                        figureSize='MNRAS_page')#, plotDensity = True, filledPlots = False,\smoothingKernel = 0, nContourLevels=3)


    fig.savefig(PlotsDir + 'pygtcPCA_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun)  + ClID + '_' + fileOut + allfiles[fileID][:-4] +'.pdf')

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

    fig.savefig(PlotsDir + 'convergencePCA_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun)  + ClID + '_'  + fileOut + allfiles[fileID][:-4] +'.pdf')