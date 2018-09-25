'''

Based on MAtt's tutorial at the ML-stat meeting

'''


import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import copy
import time
import scipy.special
import scipy.optimize
import scipy.stats

sns.set()

## Q --> approximation
## P -> actual posterior

### edgeworth expansion, KL divergence


'''
KL divergence 

ELBO --> maximaize this




VI  == any technique that gives you approximate posterior


ADVI algorithm -- Gaussian mean-field approx with batch training

Normalizing Flows


Stein VI 
'''