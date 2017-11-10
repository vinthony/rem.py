        
import numpy as np

class Criterion(object):
    def __init__(self):
        pass
    
    def __call__(self,input_data,target_data):
        return self.forward(input_data,target_data)
    
    def forward(self,input_data,target_data):
        pass
    
    def backward(self,input_data,target_data):
        pass

class CrossEntropy(Criterion):
    # cross entropy
    def __init__(self):
        self.eps = 1e-9
        super(CrossEntropy,self).__init__()
    
        
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
    
    
def L1Loss(Criterion):
    
    def __init__(self):
        super(L1Loss,self).__init__()
        
    def forward(self,input_data,target_data):
        
        return np.sum(np.abs(input_data,target_data))
    
    def backward(self,input_data,target_data):
        
        s1 = np.greater(input_data,target_data).astype('uint8')
        s_01 = np.greater(target_data,input_data).astype('uint8')
        
        return s1-s_01;
    
def L2Loss(Criterion):
    
    def __init__(self):
        super(L1Loss,self).__init__()
        
    def forward(self,input_data,target_data):
        
        return np.sum((input_data-target_data)**2)
    
    def backward(self,input_data,target_data):
        
        return 2 * (input_data - target_data)