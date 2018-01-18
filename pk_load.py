import numpy as np
print('Loading data...')


class density_profile:
    def __init__(self, data_path, para_path, test_split = 0.2, num_para = 9):
        self.data_path = data_path
        self.para_path = para_path
        self.num_para = num_para
        self.test_split = test_split


    def open_data(self):                                                        
#        with open(self.data_path) as json_data:
#            self.allData = json.load(json_data)
#
#        with open(self.para_path) as json_data:
#            self.allPara = json.load(json_data)
        
        self.allData = np.load(self.data_path)
        self.allPara = np.load(self.para_path)
        
        return self.allData, self.allPara

    def load_data(self): # randomize and split into train and test data

        allData, allPara = self.open_data()
        num_files = len(allData)                                                
        num_train = int((1-self.test_split)*num_files)

        np.random.seed(1234)
        shuffleOrder = np.arange(num_files)
        np.random.shuffle(shuffleOrder)
        allData = allData[shuffleOrder]
        allPara = allPara[shuffleOrder]
        print (allPara.shape)

        self.x_train = allData[0:num_train]
        self.y_train = allPara[0:num_train]

        self.x_test = allData[num_train:num_files]
        self.y_test = allPara[num_train:num_files]

        return (self.x_train, self.y_train), (self.x_test, self.y_test)



