import csv
import os
import numpy as np
import logging

logger = logging.getLogger()


class DataProvider(object):
    def __init__(self, name, path = 'data/', rng = None):
        self.name = name
        self.data_list = []
        self.file2id = {}
        self.id2file = []
        self.rng= rng
        if not self.rng:
            self.rng = np.random.RandomState()
        for data_file in os.listdir(path):
            if data_file == '.DS_Store':
                continue
            with open('data/' + data_file, 'rb') as f:
                reader = csv.reader(f, delimiter=',')
                tmp = list(reader)
                floats = [[float(elem) for elem in row] for row in tmp]
                data_content = np.array(floats)
                #print data_content.shape

                self.data_list.append(data_content.T)
                self.file2id[data_file] = len(self.data_list) - 1
                self.id2file.append(data_file)
        
    def getTrainingData(self):
        raise NotImplementedError
        
    def getTestData(self):
        raise NotImplementedError
        
        
class TimeWindowDataProvider(DataProvider):
    def __init__(self, timehorizon, testratio = 0.1, path = 'data/', rng = None):
        super(TimeWindowDataProvider, self).__init__('TimeWindowDataProvider', path, rng)
        # prepare data
        X = []
        for dataset in self.data_list:
            if len(dataset[0]) == 3926:
                X.append(dataset[2:6])

        x_np = np.array(X)
        #print x_np.shape
        time_series = x_np.reshape(-1, 3926, 4).swapaxes(0,1).reshape(3926,-1)

        input_data = np.zeros((x_np.shape[2] - timehorizon + 1,
                               x_np.shape[0] * x_np.shape[1] * (timehorizon - 1) \
                               + x_np.shape[0]))


        for i in range(x_np.shape[2] - timehorizon + 1):
            for j in range(timehorizon-1):
                input_data[i,x_np.shape[0] * x_np.shape[1]*j:(j+1)*x_np.shape[0] * x_np.shape[1]] = \
                                   x_np[:,:,i+j:i+j+1].reshape(-1)

            input_data[i,x_np.shape[0] * x_np.shape[1]*(timehorizon-1):] = \
                        x_np[:,0,i+timehorizon-1:i+timehorizon].reshape(-1)

        targets = np.zeros((3926, 2))
        for i in range(3926):
            if x_np[0][0][i] > x_np[0][-1][i]:
                targets[i, 0] = 1
            else:
                targets[i, 1] = 1

        targets_time = targets[timehorizon-1:]
        
        data = input_data
        targets = targets_time
        
        indizes = self.rng.permutation(data.shape[0])
        trainingIndizes = indizes[:-int(indizes.shape[0]*testratio)]
        testIndizes = indizes[-int(indizes.shape[0]*testratio):]
        
        self.trainingData = data[trainingIndizes, :]
        self.testData = data[testIndizes, :]
        
        self.trainingTargets = targets[trainingIndizes, :]
        self.testTargets = targets[testIndizes, :]
        
        
    def getTrainingData(self):
        return self.trainingData, self.trainingTargets
    
    def getTestData(self):
        return self.testData, self.testTargets

    
class TimeWindow3DDataProvider(DataProvider):
    def __init__(self, timehorizon, testratio = 0.1, path = 'data/', rng = None):
        super(TimeWindow3DDataProvider, self).__init__('TimeWindowDataProvider', path, rng)
        # prepare data
        X = []
        for dataset in self.data_list:
            if len(dataset[0]) == 3926:
                X.append(dataset[2:6])

        x_np = np.array(X)
        #print x_np.shape
        time_series = x_np.reshape(-1, 3926, 4).swapaxes(0,1).reshape(3926,-1)

        input_data = np.zeros((x_np.shape[2] - timehorizon + 1,
                               timehorizon,
                               x_np.shape[0], x_np.shape[1]))


        for i in range(x_np.shape[2] - timehorizon + 1):
            if timehorizon > 1:
                input_data[i,:-1] = x_np[:,:,i:i+timehorizon-1].swapaxes(0, 1).swapaxes(0, 2)

            input_data[i,-1, :, 0] = x_np[:,0,i+timehorizon-1]

        targets = np.zeros((3926, 2))
        for i in range(3926):
            if x_np[0][0][i] > x_np[0][-1][i]:
                targets[i, 0] = 1
            else:
                targets[i, 1] = 1

        targets_time = targets[timehorizon-1:]
        
        data = input_data
        targets = targets_time
        
        indizes = self.rng.permutation(data.shape[0])
        trainingIndizes = indizes[:-int(indizes.shape[0]*testratio)]
        testIndizes = indizes[-int(indizes.shape[0]*testratio):]
        
        self.trainingData = data[trainingIndizes, :]
        self.testData = data[testIndizes, :]
        
        self.trainingTargets = targets[trainingIndizes, :]
        self.testTargets = targets[testIndizes, :]
        
        
    def getTrainingData(self):
        return self.trainingData, self.trainingTargets
    
    def getTestData(self):
        return self.testData, self.testTargets

