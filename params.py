######################## PARAMETERS ##########################

original_dim = 2549#/2 +1  #2551 # mnist ~ 784
intermediate_dim2 = 1024#/2 #
intermediate_dim1 = 512#/2 #
intermediate_dim = 256#/2 #
latent_dim = 16

totalFiles = 512
TestFiles = 32 #128

batch_size = 8
num_epochs = 100 #110 #50
epsilon_mean = 0.0 # 1.0
epsilon_std = 1.0 # 1.0
learning_rate = 1e-3
decay_rate = 0.0

noise_factor = 0.00 # 0.0 necessary

######################## I/O #################################

DataDir = '../Cl_data/Data/'
PlotsDir = '../Cl_data/Plots/'
ModelDir = '../Cl_data/Model/'

##############################################################
