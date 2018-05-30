######################## PARAMETERS ##########################

original_dim = 256#/2 +1  #2551 # mnist ~ 784
#intermediate_dim3 = 1600
intermediate_dim2 = 192
intermediate_dim1 = 128
intermediate_dim0 = 64
intermediate_dim = 32
latent_dim = 16

ClID = ['PkNL'][0]
num_train = 256
num_test = 16
num_para = 5

batch_size = 8
num_epochs =  10000 # 20  #200 # 7500 # 200  #110 #50
epsilon_mean = 0.0 # 1.0
epsilon_std = 1e-4 ## original = 1.0, smaller the better 1e-4
learning_rate =  1e-4
decay_rate = 0.5

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
