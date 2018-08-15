import numpy as np
import corner
import params



import pygtc
import matplotlib.pylab as plt


###  TRIAL USING GEORGE -- limited range for Omega_c
ndim = 5
nwalkers = 500  # 500
nrun_burn = 100  # 300
nrun = 300  # 700


### TRIAL USING GEORGE -- extended range
# ndim = 5
# nwalkers = 600  # 400
# nrun_burn = 50  # 300
# nrun = 400  # 700



### USING GPy -- fastest, (maybe?) less precise.

## Actually shows sigma_8 -> 68pc contour
ndim = 5
nwalkers = 600  # 500
nrun_burn = 50  # 300
nrun = 300  # 700



# ndim = 5
# nwalkers = 600  # 500
# nrun_burn = 100  # 300
# nrun = 1000  # 700


# ndim = 5
# nwalkers = 1000  # 500
# nrun_burn = 50  # 300
# nrun = 2000  # 700
# fileID = 2



### USING 10k run instead of 7500
# ndim = 5
# nwalkers = 600  # 500
# nrun_burn = 50  # 300
# nrun = 600  # 700
# fileID = 2

#### Cosmological Parameters ########################################

#### Order of parameters: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s']
#        [label, true, min, max]

# param1 = ["$\Omega_c h^2$", 0.1197, 0.105, 0.155] # Actual 0.119
# param2 = ["$\Omega_b h^2$", 0.02222, 0.0215, 0.0235]
# param3 = ["$\sigma_8$", 0.829, 0.7, 0.9]
# param4 = ["$h$", 0.6731, 0.55, 0.85]
# param5 = ["$n_s$", 0.9655, 0.85, 1.05]



param1 = ["$\Omega_c h^2$", 0.1197, 0.10, 0.14] #
param2 = ["$\Omega_b h^2$", 0.02222, 0.0205, 0.0235]
param3 = ["$\sigma_8$", 0.829, 0.7, 0.9]
param4 = ["$h$", 0.6731, 0.55, 0.85]
param5 = ["$n_s$", 0.9655, 0.85, 1.05]



###################### PARAMETERS ##############################

original_dim = params.original_dim  # 2549
latent_dim = params.latent_dim  # 10

num_train = params.num_train  # 512
num_test = params.num_test  # 32
num_para = params.num_para  # 5
ClID = params.ClID


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



dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt']


fileID = 0
# samples_plotWMAP  = np.loadtxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(
#     nwalkers) + '_run' + str(nrun) + fileOut + allfiles[fileID][:-4] +'.txt')

samples_plotWMAP  = np.loadtxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(
    nwalkers) + '_run' + str(nrun)  + ClID + '_'   + fileOut + allfiles[fileID][:-4] +'.txt')

fileID = 1
# samples_plotSPT  = np.loadtxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) +
#                              '_run' + str(nrun) + fileOut + allfiles[fileID][:-4] +'.txt')
# nrun = 2000
samples_plotSPT  = np.loadtxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) +
                             '_run' + str(nrun)  + ClID + '_'  + fileOut + allfiles[fileID][:-4] +'.txt')

fileID = 2
# samples_plotPLANCK  = np.loadtxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(
#     nwalkers) +'_run' + str(nrun) + fileOut + allfiles[fileID][:-4] +'.txt')

samples_plotPLANCK  = np.loadtxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(
    nwalkers) +
                             '_run' + str(nrun)  + ClID + '_'   + fileOut + allfiles[fileID][:-4] +'.txt')




########################## Corner plots #############################

CornerPlot = True

if CornerPlot:
    fig = corner.corner(samples_plotPLANCK, labels=[param1[0], param2[0], param3[0], param4[0],
                                                  param5[0]],
                        range=[[param1[2],param1[3]], [param2[2], param2[3]], [param3[2],param3[3]],
                        [param4[2],param4[3]], [param5[2],param5[3]]],
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]],
                        show_titles=True,  title_args={"fontsize": 10})


    fig.savefig(PlotsDir +'corner_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
            nrun) + fileOut +allfiles[1][:-4] + '.pdf')




    chainLabels = ["SPT TT"]
    truthLabels = ('Planck 2015 results')

    fig = pygtc.plotGTC(samples_plotSPT, paramNames=[param1[0], param2[0], param3[0], param4[0],
                                                   param5[0]], colorsOrder= ('reds'),
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]],
                        chainLabels=chainLabels, truthLabels=truthLabels, figureSize='MNRAS_page')

    fig.savefig(PlotsDir + 'pygtc_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run'  + str(nrun) +
            fileOut + allfiles[1][:-4] +'.pdf')


chainLabels = ["PLANCK TT", "WMAP TT"]
names = [param1[0], param2[0], param3[0], param4[0], param5[0]]
truths = [param1[1], param2[1], param3[1], param4[1], param5[1]]


# Labels for the different truths
truthLabels = ( 'Planck TT+lowP 2015 results')

fig = pygtc.plotGTC( chains= [samples_plotPLANCK, samples_plotWMAP]  ,
                     colorsOrder=('greens','blues'), paramNames=names, truths=truths,
                     chainLabels=chainLabels, truthLabels=truthLabels, figureSize='MNRAS_page')#, plotDensity = True, filledPlots = False,\smoothingKernel = 0, nContourLevels=3)


fig.savefig(PlotsDir + 'pygtc_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run'  + str(nrun) +
            fileOut + allfiles[fileID][:-4] +'.pdf')


plt.show()





######## COMPARING Cl(PLANCK/SPT/WMAP) with Real data



### --------------------- mean/variance from mcmc chains ##--------------------------


def para_mcmc(samples):
    # print samples
    p1_mcmc, p2_mcmc, p3_mcmc, p4_mcmc, p5_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                       zip(*np.percentile(samples, [16, 50, 84], axis=0)))

    print('mcmc results:', p1_mcmc[0], p2_mcmc[0], p3_mcmc[0], p4_mcmc[0], p5_mcmc[0])


    return np.array([p1_mcmc, p2_mcmc, p3_mcmc, p4_mcmc, p5_mcmc])



para_mcmc(samples_plotPLANCK)


para_mcmc(samples_plotWMAP)


para_mcmc(samples_plotSPT)



### Using pre-trained GPy model #######################

ls = np.loadtxt(DataDir + 'P' + str(num_para) + 'ls_' + str(num_train) + '.txt')[2:]

normFactor = np.loadtxt(DataDir + 'normfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')
meanFactor = np.loadtxt(DataDir + 'meanfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')


from keras.models import load_model

LoadModel = True
if LoadModel:
    encoder = load_model(ModelDir + 'EncoderP' + str(num_para) + ClID + '_' + fileOut + '.hdf5')
    decoder = load_model(ModelDir + 'DecoderP' + str(num_para) + ClID + '_' + fileOut + '.hdf5')
    history = np.loadtxt(
        ModelDir + 'TrainingHistoryP' + str(num_para) + ClID + '_' + fileOut + '.txt')




import GPy


GPmodelOutfile = DataDir + 'GPy_model' + str(latent_dim) + ClID + fileOut
m1 = GPy.models.GPRegression.load_model(GPmodelOutfile + '.zip')


def GPyfit(GPmodelOutfile, para_array):


    test_pts = para_array.reshape(num_para, -1).T

    # -------------- Predict latent space ----------------------------------------

    # W_pred = np.array([np.zeros(shape=latent_dim)])
    # W_pred_var = np.array([np.zeros(shape=latent_dim)])

    m1p = m1.predict(test_pts)  # [0] is the mean and [1] the predictive
    W_pred = m1p[0]
    # W_varArray = m1p[1]


    # for j in range(latent_dim):
    #     W_pred[:, j], W_pred_var[:, j] = computedGP["fit{0}".format(j)].predict(encoded_xtrain[j],
    #                                                                             test_pts)

    # -------------- Decode from latent space --------------------------------------

    x_decoded = decoder.predict(W_pred.reshape(latent_dim, -1).T )

    return (normFactor * x_decoded[0]) + meanFactor



########## REAL DATA with ERRORS #############################
# Planck/SPT/WMAP data
# TE, EE, BB next

dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt']

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



yPLANCK = para_mcmc(samples_plotPLANCK)

x_decodedGPy = GPyfit(GPmodelOutfile, yPLANCK[:,0])


yPLANCK2015 = np.array([param1[1], param2[1], param3[1], param4[1], param5[1]])

x_decodedGPy2015 = GPyfit(GPmodelOutfile, yPLANCK2015)





plt.figure(322, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(1, 1, height_ratios=[1])
gs.update(hspace=0.02, left=0.2, bottom = 0.15)  # set the spacing between axes.
ax0 = plt.subplot(gs[0])


ax0.errorbar(l, Cl, yerr = [emin, emax], ecolor = 'k', alpha = 0.1, label = 'PLANCK Data')
ax0.plot(ls, x_decodedGPy, 'r-', label = 'Best estimate')
ax0.plot(ls, x_decodedGPy2015, 'b-', label = 'Planck2015 estimate')



ax0.set_ylabel(r'$l(l+1)C_l/2\pi [\mu K^2]$')
ax0.set_xlabel(r'$l$')

ax0.legend()

# plt.show()

# ax0.set_xscale('log')
# ax0.set_yscale('log')
