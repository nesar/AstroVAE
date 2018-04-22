#!/usr/bin/env python

import numpy as np
import emcee

'''
MCMC fitting template. 
This template fits a 1-d gaussian, if you 
figure out how to use it for more complicated distributions
I'd appreciate if you let me know :)
banados@mpia.de
'''

#First let's create a gaussian data

data = np.random.normal(loc=100.0, scale=3.0, size=1000)

# Then, define the probability distribution that you would like to sample.
def lnprob(p, vec):
    diff = vec-p[0]
    N = len(vec)
    return -0.5 * N * np.log(2 * np.pi) - N * np.log(p[1]) - 0.5 \
                                    * np.sum(( (vec - p[0]) / p[1] ) ** 2)
    
               
# We'll sample a Gaussian which has 2 parameters: mean and sigma...
ndim = 2

# We'll sample with 250 walkers. (nwalkers must be an even number)
nwalkers = 250

# Choose an initial set of positions for the walkers.
p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]

# Initialize the sampler with the chosen specs.
#The "a" parameter controls the step size, the default is a=2,
#but in this case works better with a=4 see below or page 10 in the paper
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data], a=4)

# Run 200 steps as a burn-in.
print "Burning in ..."
pos, prob, state = sampler.run_mcmc(p0, 200)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, sample for 1000
# steps. (rstate0 is the state of the internal random number generator)
print "Running MCMC ..."
pos, prob, state = sampler.run_mcmc(pos, 1000, rstate0=state)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 250-dimensional
# vector.
af = sampler.acceptance_fraction
print "Mean acceptance fraction:", np.mean(af)
af_msg = '''As a rule of thumb, the acceptance fraction (af) should be 
                            between 0.2 and 0.5
            If af < 0.2 decrease the a parameter
            If af > 0.5 increase the a parameter
            '''

print af_msg

# If you have installed acor (http://github.com/dfm/acor), you can estimate
# the autocorrelation time for the chain. The autocorrelation time is also
# a vector with 10 entries (one for each dimension of parameter space).
try:
    print "Autocorrelation time:", sampler.acor
except ImportError:
    print "You can install acor: http://github.com/dfm/acor"
    
maxprob_indice = np.argmax(prob)

mean_fit, sigma_fit = pos[maxprob_indice]

print "Estimated parameters: mean, sigma = %f, %f" % (mean_fit, sigma_fit)

mean_samples = sampler.flatchain[:,0]
sigma_samples = sampler.flatchain[:,1]

mean_std = mean_samples.std()
sigma_std = sigma_samples.std()

print "parameters' error: mean, sigma = %f, %f" % (mean_std, sigma_std)

# Finally, you can plot the projected histograms of the samples using
# matplotlib as follows (as long as you have it installed).
try:
    import matplotlib.pyplot as plt
except ImportError:
    print "Try installing matplotlib to generate some sweet plots..."
else:
    plt.hist(mean_samples, 100)
    plt.title("Samples for mean")
    plt.show()
    plt.title("Samples for sigma")
    plt.hist(sigma_samples, 100)
    plt.show()