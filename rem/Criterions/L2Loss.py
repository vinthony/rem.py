import numpy as np
from .Criterion import Criterion

class L2Loss(Criterion):
    
    def __init__(self):
        super(L1Loss,self).__init__()
        
    def forward(self,input_data,target_data):
        
        return np.sum((input_data-target_data)**2)
    
    def backward(self,input_data,target_data):
        
        return 2 * (input_data - target_data)