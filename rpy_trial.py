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

y_train = np.array(r('y_train2'))
u_test = np.array(r('u_test2'))

print y_train.shape

########



DiceKriging = importr('DiceKriging')

r('require(foreach)')

# below an example for a computer with 2 cores, but also work with 1 core

# r('multistart <- 4')

# r('nCores <- multistart')
# r('require(doParallel)')
# r('cl <-  makeCluster(nCores)')
# r('registerDoParallel(cl)')
r('mods8 <- km(~., design = u_train, response = s_train, multistart = multistart)')
# r('stopCluster(cl)')