import numpy as np
from .Criterion import Criterion


class L1Loss(Criterion):
    
    def __init__(self):
        super(L1Loss,self).__init__()
        
    def forward(self,input_data,target_data):
        
        return np.sum(np.abs(input_data,target_data))
    
    def backward(self,input_data,target_data):
        
        s1 = np.greater(input_data,target_data).astype('uint8')
        s_01 = np.greater(target_data,input_data).astype('uint8')
        
        return s1-s_01;