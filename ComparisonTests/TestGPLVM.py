import pylab as pb
import numpy as np
import GPy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# pb.ion()

Utrain = np.loadtxt("/home/mickael/Documents/GitProjects/sandbox/AstroVAE/Cl_data/Data/LatinCosmoP51024.txt")
Ytrain = np.load("/home/mickael/Documents/GitProjects/sandbox/AstroVAE/Cl_data/Data/LatinCl_1024Mod.npy")

# For consistency, use the s_input^2/s_8_output^2 transform
Strain = np.load("/home/mickael/Documents/GitProjects/sandbox/AstroVAE/Cl_data/Data/LatinCl_1024ModSigma8.npy")
Ytrain = np.dot(np.diag((Utrain[:,2]/Strain[:,0])*(Utrain[:,2]/Strain[:,0])), Ytrain)

# Initial test data (with no Strain)
# Ytest = np.load("/home/mickael/Documents/GitProjects/sandbox/AstroVAE/Cl_data/Data/LatinCl_25Mod.npy")
# Utest = np.loadtxt("/home/mickael/Documents/GitProjects/sandbox/AstroVAE/Cl_data/Data/LatinCosmoP525.txt")

# New test data (with transform)
Utest = np.loadtxt("/home/mickael/Documents/GitProjects/AstroVAE/ComparisonTests/params.txt")
Ytest = np.loadtxt("/home/mickael/Documents/GitProjects/AstroVAE/ComparisonTests/TTtrue.txt")

# Remove first 10 observations (not pertinent)
Ytrain = Ytrain[:, 10:Ytrain.shape[1]]
Ytest = Ytest[:, 8:Ytest.shape[1]] # Already done for new data

# Max latent dimension
input_dim = 10

m_gplvm = GPy.models.BayesianGPLVM(Y=Ytrain, input_dim=input_dim, num_inducing=100)
# m_gplvm.Gaussian_noise.variance.constrain_fixed(10)

m_gplvm.optimize(messages=1, max_iters=5e5)

v1 = m_gplvm.plot_latent()

m_gplvm.kern.plot_ARD()

# Get latent coordinates (slight difference with the latent plot?)
Xtest = m_gplvm.infer_newX(Ytest)
Xtrain = m_gplvm.infer_newX(Ytrain)

# Nesar's special
plt.figure()
plt.scatter(Xtrain[0].mean[:,0], Xtrain[0].mean[:,1], c = Utrain[:,2])
plt.plot(Xtest[0].mean[:,0], Xtest[0].mean[:,1], 'ro')

# Get prediction from latent space coordinates
Xt = m_gplvm.predict(np.ones((2, input_dim)))
Ytrain_r = m_gplvm.predict(Xtrain[0].mean)

plt.figure()
plt.plot(Ytrain_r[0][0,:])
plt.plot(Ytrain[0,:], 'r')

# Model for latent coordinates
m1 = GPy.models.GPRegression(Utrain, np.reshape(Xtrain[0].mean[:,0], [-1,1]).values)
m1.Gaussian_noise.variance.constrain_fixed(0.00001)
m1.optimize()
print(m1)

m2 = GPy.models.GPRegression(Utrain, np.reshape(Xtrain[0].mean[:,1], [-1,1]).values)
m2.Gaussian_noise.variance.constrain_fixed(0.00001)
m2.optimize()
print(m2)

m3 = GPy.models.GPRegression(Utrain, np.reshape(Xtrain[0].mean[:,2], [-1,1]).values)
m3.Gaussian_noise.variance.constrain_fixed(0.00001)
m3.optimize()
print(m3)

m4 = GPy.models.GPRegression(Utrain, np.reshape(Xtrain[0].mean[:,3], [-1,1]).values)
m4.Gaussian_noise.variance.constrain_fixed(0.00001)
m4.optimize()
print(m4)

m5 = GPy.models.GPRegression(Utrain, np.reshape(Xtrain[0].mean[:,4], [-1,1]).values)
m5.Gaussian_noise.variance.constrain_fixed(0.00001)
m5.optimize()
print(m5)

m6 = GPy.models.GPRegression(Utrain, np.reshape(Xtrain[0].mean[:,5], [-1,1]).values)
m6.Gaussian_noise.variance.constrain_fixed(0.00001)
m6.optimize()
print(m6)

m7 = GPy.models.GPRegression(Utrain, np.reshape(Xtrain[0].mean[:,6], [-1,1]).values)
m7.Gaussian_noise.variance.constrain_fixed(0.00001)
m7.optimize()
print(m7)

m8 = GPy.models.GPRegression(Utrain, np.reshape(Xtrain[0].mean[:,7], [-1,1]).values)
m8.Gaussian_noise.variance.constrain_fixed(0.00001)
m8.optimize()
print(m8)

m9 = GPy.models.GPRegression(Utrain, np.reshape(Xtrain[0].mean[:,8], [-1,1]).values)
m9.Gaussian_noise.variance.constrain_fixed(0.00001)
m9.optimize()
print(m9)

m10 = GPy.models.GPRegression(Utrain, np.reshape(Xtrain[0].mean[:,9], [-1,1]).values)
m10.Gaussian_noise.variance.constrain_fixed(0.00001)
m10.optimize()
print(m10)


# Now reconstruct:
Xtest_p1 = m1.predict(Utest)[0]
Xtest_p2 = m2.predict(Utest)[0]
Xtest_p3 = m3.predict(Utest)[0]
Xtest_p4 = m4.predict(Utest)[0]
Xtest_p5 = m5.predict(Utest)[0]
Xtest_p6 = m6.predict(Utest)[0]
Xtest_p7 = m7.predict(Utest)[0]
Xtest_p8 = m8.predict(Utest)[0]
Xtest_p9 = m9.predict(Utest)[0]
Xtest_p10 = m10.predict(Utest)[0]

Xtest_p = np.hstack([Xtest_p1, Xtest_p2, Xtest_p3, Xtest_p4,
                     Xtest_p5, Xtest_p6, Xtest_p7, Xtest_p8, #np.zeros([25,4])])
                     Xtest_p9, Xtest_p10])

Ytest_r = m_gplvm.predict(Xtest_p)

plt.figure()
plt.plot(Ytest_r[0][0,:])
plt.plot(Ytest[0,:], 'r')

Xtest_p[0,]
print(Xtest[0].mean[0,])