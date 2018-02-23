######################## PARAMETERS ##########################

original_dim = 2549#/2 +1  #2551 # mnist ~ 784
intermediate_dim2 = 1024#/2 #
intermediate_dim1 = 512#/2 #
intermediate_dim = 256#/2 #
latent_dim = 20

num_train = 256
num_test = 32 #128
num_para = 5

batch_size = 8
num_epochs = 200 #110 #50
epsilon_mean = 0.0 # 1.0
epsilon_std = 1e-4 ## original = 1.0, smaller the better 1e-4
learning_rate =  1e-4
decay_rate = 1.0

noise_factor = 0.0 # 0.0 necessary

######################## I/O #################################

DataDir = '../Cl_data/Data/'
PlotsDir = '../Cl_data/Plots/'
ModelDir = '../Cl_data/Model/'

fileOut = 'P'+str(num_para)+'Model_tot' + str(num_train) + '_batch' + str(batch_size) + '_lr' + str(
    learning_rate) + '_decay' + str(decay_rate) + '_z' + str(latent_dim) + '_epoch' + str(
    num_epochs)

##############################################################
