"""
http://heather.cs.ucdavis.edu/~matloff/rpy2.html

>>> from rpy2.robjects import r
>>> r('x <- rnorm(100)')  # generate x at R
>>> r('y <- x + rnorm(100,sd=0.5)')  # generate y at R
>>> r('plot(x,y)')  # have R plot them
>>> r('lmout <- lm(y~x)')  # run the regression
>>> r('print(lmout)')  # print from R
>>> loclmout = r('lmout') # download lmout from R to Python
>>> print loclmout  # print locally
>>> print loclmout.r['coefficients']  # print one component
Now let's apply some R operations to some Python variables:

>>> u = range(10)  # set up another scatter plot, this one local
>>> e = 5*[0.25,-0.25]
>>> v = u[:]
>>> for i in range(10): v[i] += e[i]
>>> r.plot(u,v)
>>> r.assign('remoteu',u)  # ship local u to R
>>> r.assign('remotev',v)  # ship local v to R
>>> r('plot(remoteu,remotev)')  # plot there
There are many more functions. See the RPy documentation for details.

"""

import numpy as np
from rpy2.robjects import r
from rpy2.robjects.packages import importr

RcppCNPy = importr('RcppCNPy')
# RcppCNPy.chooseCRANmirror(ind=1) # select the first mirror in the list

########

# importr('data.table')#, lib='~/R-lib') # for first file reading


########


# library(RcppCNPy)

################################# I/O #################################

# Note that the 3rd variable is not used here, and the first two points of the spectrum can be removed
r('u_train2 <- as.matrix(read.csv("../Cl_data/Data/LatinCosmoP51024.txt", sep = " ", header = '
  'F))') ## training design
# r('s_train <- npyLoad("../Cl_data/Data/LatinCl_1024ModSigma8.npy")') ## training sigma8 output
# r('y_train2 <- diag(drop(u_train2[,3]^2 / s_train^2))') ## %*% y_train



r('y_train2 <- as.matrix(read.csv("../Cl_data/Data/P5TTCl_1024.txt", sep = " ", header = ''F))[,'
  '-(1:7)]')



r('u_test2 <- as.matrix(read.csv("ComparisonTests/VAE_data/params.txt", sep = " ", header = F))') #
# testing design
r('y_test2 <- as.matrix(read.csv("ComparisonTests/VAE_data/TTtrue.txt", sep = " ", header = F))') #[,-c(1,2)] # testing spectrum curves

r('matplot(t(y_train2), type = "l")')


########


########################### PCA ###################################


Dicekriging = importr('DiceKriging')

r('require(foreach)')

r('svd(y_train2)')



r('nrankmax <- 3')   ## Number of components


r('svd_decomp2 <- svd(y_train2)')
r('svd_weights2 <- svd_decomp2$u[, 1:nrankmax] %*% diag(svd_decomp2$d[1:nrankmax])')



########################### GP train #################################

## Build GP models
GPareto = importr('GPareto')


r('models_svd2 <- list()')



r('''for (i in 1: nrankmax){
        mod_s <- km(~., design = u_train2, response = svd_weights2[, i])
        models_svd2 <- c(models_svd2, list(mod_s))
                          }''')
# stopCluster(cl)

######################### INFERENCE ########################


r('wtestsvd2 <- predict_kms(models_svd2, newdata = u_test2, type = "UK")')
r('reconst_s2 <- t(wtestsvd2$mean) %*% t(svd_decomp2$v[,1:nrankmax])')


##################### TESTING, ratio test ######################

r('plot(reconst_s2[1,]/y_test2[1,], ylim = c(0.99, 1.01), type = "l", xlab = "l", ylab = "predicted/real") ')
r('for(i in 1:25){lines(reconst_s2[i,]/y_test2[i,])} ')
r('abline(h = 1.005) ')
r('abline(h = 0.995) ')



##################### Weights vs max ratio #######################

r('max_err_rat2 <- rep(NA, nrankmax)')
r(''' for(i in 1:nrankmax){
  tmp <- t(wtestsvd2$mean)[,1:i,drop = F] %*% t(svd_decomp2$v[,1:nrankmax])[1:i,,drop = F]
  max_err_rat2[i] <- max(abs(tmp/y_test2 - 1))
} ''')
r('plot(max_err_rat2, xlab = "Number of PCA components", ylab = "Max of deviation of ypred/ytest")')
r('lines(max_err_rat2, lty = 2)')
r('abline(h = 0.5)')


################ Not sure what this is ###############
# r('plot(max_err_rat - max_err_rat2)')



################ npy - rpy connection ################

import rpy2.robjects.numpy2ri

from rpy2.robjects.numpy2ri import numpy2ri

rpy2.robjects.numpy2ri.activate()

import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
ro.conversion.py2ri = numpy2ri


# r.assign('u_test3',u_test[0,:])

# y_train = np.array(r('y_train2'))
# print y_train.shape


u_test = r('u_test2')
r.assign('u_test3',u_test)


r('wtestsvd2 <- predict_kms(models_svd2, newdata = u_test3 , type = "UK")')

w_test = np.array(r('wtestsvd2'))
