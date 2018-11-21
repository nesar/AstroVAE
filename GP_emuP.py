"""
Requires the following installations:

1. R (R studio is the easiest option: https://www.rstudio.com/products/rstudio/download/).
Installing R packages is easy, in R studio, command install.packages("package_name") works
(https://www.dummies.com/programming/r/how-to-install-load-and-unload-packages-in-r/)
The following R packages are required:
    1a. RcppCNPy
    1b. DiceKriging
    1c. GPareto

2. rpy2 -- which runs R under the hood (pip install rpy2 should work)
# http://rpy.sourceforge.net/rpy2/doc-2.1/html/index.html

3. astropy for reading fits files

"""


##### Generic packages ###############
import numpy as np
import matplotlib.pylab as plt
import time

###### astropy for fits reading #######
from astropy.io import fits as pf
import astropy.table

###### R kernel imports from rpy2 #####
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r
from rpy2.robjects.packages import importr


############################# PARAMETERS ##############################

fitsfileIn = "../P_data/2ndpass_vals_for_test.fits"
nRankMax = 32
ro.r.assign("nrankmax", nRankMax)

################################# I/O #################################

RcppCNPy = importr('RcppCNPy')
# RcppCNPy.chooseCRANmirror(ind=1) # select the first mirror in the list


Allfits = pf.open(fitsfileIn)
AllData = astropy.table.Table(Allfits[1].data)

parameter_array = np.array([AllData['RHO'], AllData['SIGMA_LAMBDA'], AllData['TAU'],
                            AllData['SSPT']]).T

nr, nc = parameter_array.shape
u_train = ro.r.matrix(parameter_array, nrow=nr, ncol=nc)

ro.r.assign("u_train2", u_train)
r('dim(u_train2)')

pvec = (AllData['PVEC'])  # .newbyteorder('S')
# print(  np.unique( np.argwhere( np.isnan(pvec) )[:,0]) )

np.savetxt('pvec.txt', pvec)
pvec = np.loadtxt('pvec.txt')

nr, nc = pvec.shape
y_train = ro.r.matrix(pvec, nrow=nr, ncol=nc)

ro.r.assign("y_train2", y_train)
r('dim(y_train2)')

########################### PCA ###################################

def PCA():
    Dicekriging = importr('DiceKriging')

    r('require(foreach)')

    r('svd(y_train2)')

    r('svd_decomp2 <- svd(y_train2)')
    r('svd_weights2 <- svd_decomp2$u[, 1:nrankmax] %*% diag(svd_decomp2$d[1:nrankmax])')

######################## GP FITTING ################################

## Build GP models
# This is evaluated only once for the file name. GP fitting is not required if the file exists.

def GP():
    GPareto = importr('GPareto')

    # GPmodel = '"R_GP_model.RData"'  ## Double and single quotes are necessary
    #
    # ro.r('''
    #
    # GPmodel <- gsub("to", "",''' + GPmodel + ''')
    #
    # ''')

    r('''if(file.exists("R_GP_models.RData")){
            load("R_GP_models.RData")
        }else{
            models_svd2 <- list()
            for (i in 1: nrankmax){
                mod_s <- km(~., design = u_train2, response = svd_weights2[, i])
                models_svd2 <- c(models_svd2, list(mod_s))
            }
            save(models_svd2, file = "R_GP_models.RData")

         }''')

    r('''''')



PCA()
GP()

######################### INFERENCE ##################################




def GP_fit(para_array):
    para_array = np.expand_dims(para_array, axis=0)

    nr, nc = para_array.shape
    Br = ro.r.matrix(para_array, nrow=nr, ncol=nc)

    ro.r.assign("Br", Br)

    r('wtestsvd2 <- predict_kms(models_svd2, newdata = Br , type = "UK")')
    r('reconst_s2 <- t(wtestsvd2$mean) %*% t(svd_decomp2$v[,1:nrankmax])')

    y_recon = np.array(r('reconst_s2'))

    return y_recon[0]



##################################### TESTING ##################################


plt.rc('text', usetex=True)   # Slower
plt.rc('font', size=12)  # 18 usually

plt.figure(999, figsize=(7, 6))
from matplotlib import gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(hspace=0.02, left=0.2, bottom=0.15)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_ylabel(r'$P(x)$')

ax1.axhline(y=1, ls='dotted')
# ax1.axhline(y=-1e-6, ls='dashed')
# ax1.axhline(y=1e-6, ls='dashed')

ax1.set_xlabel(r'$x$')

ax1.set_ylabel(r'emu/real - 1')
ax1.set_ylim(-1e-5, 1e-5)


for x_id in [3, 23, 43, 64, 93, 109, 11]:
    x_decodedGPy = GP_fit(parameter_array[x_id])  ## input parameters
    x_test = pvec[x_id]

    ax0.plot(x_decodedGPy, alpha=1.0, ls='--', label='emu')
    ax0.plot(x_test, alpha=0.9, label='real')
    plt.legend()

    ax1.plot(x_decodedGPy[1:] / x_test[1:] - 1)

plt.show()



######### TEMPLATE FOR MCMC LIKELIHOOD FUNCTION #######################
# For emcee

def lnlike(theta, x, y, yerr):
    p1, p2, p3, p4, p5 = theta

    new_params = np.array([p1, p2, p3, p4, p5])

    model = GP_fit(new_params)
    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model) / yerr) ** 2.))


