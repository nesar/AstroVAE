######################## PARAMETERS ##########################

original_dim = 256#/2 +1  #2551 # mnist ~ 784
#intermediate_dim3 = 1600
intermediate_dim2 = 512
intermediate_dim1 = 256
intermediate_dim0 = 128
intermediate_dim = 64
latent_dim = 32

ClID = ['PkNL'][0]
num_train = 512
num_test = 16
num_para = 5

batch_size = 16
num_epochs =  10000 # 20  #200 # 7500 # 200  #110 #50
epsilon_mean = 0.0 # 1.0
epsilon_std = 1e-6#1e-6 ## original = 1.0, smaller the better 1e-4
learning_rate =  1e-5
decay_rate = 0.1

noise_factor = 0.0 # 0.0 necessary

######################## I/O #################################
MainDir = '../Pk_data/'

DataDir = MainDir+'Data/'
PlotsDir = MainDir+'Plots/'
ModelDir = MainDir+'Model/'

fileOut = 'P'+str(num_para)+'Model_tot' + str(num_train) + '_batch' + str(batch_size) + '_lr' + str(
    learning_rate) + '_decay' + str(decay_rate) + '_z' + str(latent_dim) + '_epoch' + str(
    num_epochs)

##############################################################
