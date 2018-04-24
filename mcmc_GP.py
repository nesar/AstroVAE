import numpy as np
import matplotlib.pylab as plt


# Choose the "true" parameters.

# cosmological parameters here

m_true = -0.9594
b_true = 4.294
f_true = 0.534


########## REAL DATA with ERRORS #############################
# Generate some synthetic data from the model.


N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)

allfiles = ['./BICEP.txt', './WMAP.txt', './SPTpol.txt', './PLANCKlegacy.txt']

dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt']

# lID = np.array([1, 0, 2, 0])
# ClID = np.array([1, 1, 3, 1])
# emaxID = np.array([1, 2, 4, 2])
# eminID = np.array([1, 2, 4, 2])


lID = np.array([0, 2, 0])
ClID = np.array([1, 3, 1])
emaxID = np.array([2, 4, 2])
eminID = np.array([2, 4, 2])

print allfiles

import numpy as np

for fileID in [2]:
    with open(dirIn + allfiles[fileID]) as f:
        lines = (line for line in f if not line.startswith('#'))
        allCl = np.loadtxt(lines, skiprows=1)

    l = allCl[:, lID[fileID]]
    Cl = allCl[:, ClID[fileID]]
    emax = allCl[:, emaxID[fileID]]
    emin = allCl[:, eminID[fileID]]

    print l.shape

x = l
y = Cl
yerr = emax


######### LEAST SQUARE #######################################

# A = np.vstack((np.ones_like(x), x)).T
# C = np.diag(yerr * yerr)
# cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
# b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

#############################################################


######### MAX LIKELIHOOD ESTIMATION #######################

def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


import scipy.optimize as op
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result["x"]


#############################################################


######### MCMC #######################

# log likelihood function -- to be replaced by the emulator as a function of cosmological parameters
# def lnlike(theta, x, y, yerr):
#     m, b, lnf = theta
#     model = m * x + b
#     inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
#     return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


# Use flat prior ?
def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

# Dunno what to do here  -- just a combination of likelihood and prior?
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)



ndim, nwalkers = 3, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]



import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

sampler.run_mcmc(pos, 500)


samples = sampler.chain[:, 50:, :].reshape((-1, ndim))



####### CORNER PLOT ESTIMATES #######################################


import corner
fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
                      truths=[m_true, b_true, np.log(f_true)])
fig.savefig("triangle.png")


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
