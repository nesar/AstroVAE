######################## EXTENDED ell ~10k models ##########################

original_dim = 9999 #/2 +1  #2551 # mnist ~ 784
intermediate_dim3 = 2048
intermediate_dim2 = 1024
intermediate_dim1 = 512
intermediate_dim0 = 256
intermediate_dim = 128
latent_dim = 32

ClID = ['TT', 'EE', 'BB', 'TE'][0]
num_train = 1024
num_test = 32
num_para = 10

batch_size = 4
num_epochs =  100 #100 # 8000 #7500 # 20  #200 # 7500 # 200  #110 #50
epsilon_mean = 0.0 # 1.0
epsilon_std = 1e-4 ## original = 1.0, smaller the better 1e-4
learning_rate =  1e-4
decay_rate = 1.0

noise_factor = 0.0 # 0.0 necessary

######################## I/O #################################
MainDir = '../Cl_data/'

DataDir = MainDir+'Data/Extended/Extended'
PlotsDir = MainDir+'Plots/Extended/'
ModelDir = MainDir+'Model/Extended/'

fileOut = 'P'+str(num_para)+'Model_tot' + str(num_train) + '_batch' + str(batch_size) + '_lr' + \
          str(
    learning_rate) + '_decay' + str(decay_rate) + '_z' + str(latent_dim) + '_epoch' + str(
    num_epochs)

##############################################################

######################## EXTENDED ell ~10k models ##########################

original_dim = 9999 #/2 +1  #2551 # mnist ~ 784
intermediate_dim3 = 2048
intermediate_dim2 = 1024
intermediate_dim1 = 512
intermediate_dim0 = 256
intermediate_dim = 128
latent_dim = 64

ClID = ['TT', 'EE', 'BB', 'TE'][0]
num_train = 1024
num_test = 32
num_para = 10

batch_size = 8
num_epochs =  500 #100 # 8000 #7500 # 20  #200 # 7500 # 200  #110 #50
epsilon_mean = 0.0 # 1.0
epsilon_std = 1e-4 ## original = 1.0, smaller the better 1e-4
learning_rate =  1e-3
decay_rate = 1.0

noise_factor = 0.0 # 0.0 necessary

######################## I/O #################################
MainDir = '../Cl_data/'

DataDir = MainDir+'Data/Extended/Extended'
PlotsDir = MainDir+'Plots/Extended/'
ModelDir = MainDir+'Model/Extended/'

fileOut = 'P'+str(num_para)+'Model_tot' + str(num_train) + '_batch' + str(batch_size) + '_lr' + \
          str(
    learning_rate) + '_decay' + str(decay_rate) + '_z' + str(latent_dim) + '_epoch' + str(
    num_epochs)

#############



######################## EXTENDED ell ~10k models ##########################

original_dim = 9999 #/2 +1  #2551 # mnist ~ 784
intermediate_dim3 = 2048
intermediate_dim2 = 1024
intermediate_dim1 = 512
intermediate_dim0 = 256
intermediate_dim = 128
latent_dim = 64

ClID = ['TT', 'EE', 'BB', 'TE'][0]
num_train = 1024
num_test = 32
num_para = 10

batch_size = 8
num_epochs =  50 #100 # 8000 #7500 # 20  #200 # 7500 # 200  #110 #50
epsilon_mean = 0.0 # 1.0
epsilon_std = 1e-4 ## original = 1.0, smaller the better 1e-4
learning_rate =  1e-3
decay_rate = 1.0

noise_factor = 0.0 # 0.0 necessary

######################## I/O #################################
MainDir = '../Cl_data/'

DataDir = MainDir+'Data/Extended/Extended'
PlotsDir = MainDir+'Plots/Extended/'
ModelDir = MainDir+'Model/Extended/'

fileOut = 'P'+str(num_para)+'Model_tot' + str(num_train) + '_batch' + str(batch_size) + '_lr' + \
          str(
    learning_rate) + '_decay' + str(decay_rate) + '_z' + str(latent_dim) + '_epoch' + str(
    num_epochs)

#############