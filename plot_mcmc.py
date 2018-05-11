import numpy as np
import corner
import params



import pygtc

ndim = 5
nwalkers = 500  # 500
nrun_burn = 100  # 300
nrun = 300  # 700
fileID = 0

#### Cosmological Parameters ########################################

#### Order of parameters: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s']
#        [label, true, min, max]

param1 = ["$\Omega_c h^2$", 0.1197, 0.105, 0.155] # Actual 0.119
param2 = ["$\Omega_b h^2$", 0.02222, 0.0215, 0.0235]
param3 = ["$\sigma_8$", 0.829, 0.7, 0.9]
param4 = ["$h$", 0.6731, 0.55, 0.85]
param5 = ["$n_s$", 0.9655, 0.85, 1.05]

###################### PARAMETERS ##############################

original_dim = params.original_dim  # 2549
latent_dim = params.latent_dim  # 10

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



dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt']


fileID = 0
samples_plotWMAP  = np.loadtxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(
    nwalkers) +
                             '_run' + str(nrun) + fileOut + allfiles[fileID][:-4] +'.txt')

fileID = 1
samples_plotSPT  = np.loadtxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) +
                             '_run' + str(nrun) + fileOut + allfiles[fileID][:-4] +'.txt')

fileID = 2
samples_plotPLANCK  = np.loadtxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(
    nwalkers) +
                             '_run' + str(nrun) + fileOut + allfiles[fileID][:-4] +'.txt')


CornerPlot = True

if CornerPlot:
    fig = corner.corner(samples_plotSPT, labels=[param1[0], param2[0], param3[0], param4[0],
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
truthLabels = ( 'Planck TT-lowP 2015 results')

fig = pygtc.plotGTC( chains= [samples_plotPLANCK, samples_plotWMAP]  ,
                     colorsOrder=('greens','blues'), paramNames=names, truths=truths,
                     chainLabels=chainLabels, truthLabels=truthLabels, figureSize='MNRAS_page')#, plotDensity = True, filledPlots = False,\smoothingKernel = 0, nContourLevels=3)


fig.savefig(PlotsDir + 'pygtc_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run'  + str(nrun) +
            fileOut + allfiles[fileID][:-4] +'.pdf')