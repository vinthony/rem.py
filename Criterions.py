        
import numpy as np

class CrossEntropy(object):
    # cross entropy
    def __init__(self):
        self.eps = 1e-9
        pass
    def __call__(self,input_data,target_data):
        return self.forward(input_data,target_data)
        
    def forward(self,input_data,target_data):
        # bs x 10
        ee = np.exp(input_data) 
        sf = ee / np.sum(ee,axis=1).reshape((-1,1))

        self.softmax = sf;
        
        self.output = np.mean(np.sum(-target_data * np.log(sf+self.eps),axis=1))
        
        return self.output
        
    def backward(self,input_data,target_data):
        # \sum( - t * log(exp(i)/ \sum(exp(i)) )) 
        # batchsize * 10, 

        return self.softmax - target_data