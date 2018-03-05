import numpy as np
print('Loading data...')

# x_train = density profile.

m_particle = 3.65235543e10


class density_profile:
    def __init__(self, data_path, para_path1, para_path2, test_split = 0.2, num_para = 1):
        self.data_path = data_path
        self.para_path1 = para_path1
        self.para_path2 = para_path2
        self.num_para = num_para
        self.test_split = test_split


    def open_data(self):                                                        
#        with open(self.data_path) as json_data:
#            self.allData = json.load(json_data)
#
#        with open(self.para_path) as json_data:
#            self.allPara = json.load(json_data)
        
        self.allData = np.load(self.data_path)
        self.allPara1 = np.load(self.para_path1)
        self.allPara2 = np.load(self.para_path2)
        
        return self.allData, self.allPara1, self.allPara2

    def load_data(self): # randomize and split into train and test data

        allData, allPara1, allPara2 = self.open_data()
        num_files = len(allData)                                                
        num_train = int((1-self.test_split)*num_files)

        np.random.seed(1234)
        shuffleOrder = np.arange(num_files)
        np.random.shuffle(shuffleOrder)
        allData = allData[shuffleOrder]/(1e4*m_particle)
        allPara1 = allPara1[shuffleOrder]/(1e4*m_particle)
        allPara2 = allPara2[shuffleOrder]
        allPara = np.dstack((allPara1, allPara2))[0]
        print (allPara.shape)

        self.x_train = allData[0:num_train]
        self.y_train = allPara[0:num_train]

        self.x_test = allData[num_train:num_files]
        self.y_test = allPara[num_train:num_files]

        return (self.x_train, self.y_train), (self.x_test, self.y_test)



