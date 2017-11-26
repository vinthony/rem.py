import numpy as np
from rem.Criterions.Criterion import Criterion

class CrossEntropy(Criterion):
    # cross entropy
    def __init__(self):
        self.eps = 1e-9
        super(CrossEntropy,self).__init__()
    
        
    def forward(self,input_data,target_data):
        # bs x 10
        input_data = input_data - np.expand_dims(np.max(input_data,axis=1),axis=1)
        ee = np.exp(input_data) 
        sf = ee / (np.sum(ee,axis=1).reshape((-1,1))+self.eps)

        self.softmax = sf;
        
        self.output = np.mean(np.sum(-target_data * np.log(sf+self.eps),axis=1))
        
        return self.output
        
    def backward(self,input_data,target_data):
        # \sum( - t * log(exp(i)/ \sum(exp(i)) )) 
        # batchsize * 10, 

        return self.softmax - target_data