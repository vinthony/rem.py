# -*- coding: utf-8 -*-
"""
Spyder Editor 2017年11月2日23:41:00

This is a temporary script file.
"""

import json
import numpy as np

from Layers import Conv2d,NonLinear,BN,Linear

def data_loader():
    # generater an image from datasert
    input_image, label = get_batch()
    
    yield input_image,label
    
    
    

class Network(Object):
    def __init__(self,**kwarg):
         self.conv1 = Conv2d(3,64,3,3,1,1,1,1);
         self.bn1 = BN(64);
         self.maxpool = Maxpool(2); # 128x14x14
         self.conv2 = Conv2d(64,64,3,3,1,1,1,1);
         self.bn2 = BN();
         self.averagepool = Averagepool(2); # 128x7x7
         self.relu  = NonLinear(sub_type='relu');
         self.softmax = NonLinear(sub_type='softmax');
         self.linear1 = Linear(49);
         self.linear2 = Linear(10);
         self.is_init = False
         self.stack = []
         
    def __call__(self,input_data):
        return self.forward(input_data)
        
    def forward(self,input_data):
        
        x1 = self.bn1(self.relu(self.conv1(input_data)))
        
        x2 = self.bn2(self.relu(self.conv2(x1)))
        
        x3 = self.softmax(self.linear1(self.linear2(x2)))
        
        if not self.is_init:
            # FIND THE OBJECT WHICH IS THE INSTANCE OF lAYER
            self.get_stack();
            self.is_init = True            
        
        return x3
        
    
 ###               
 ###   def backward(self,input_data,gradient_data):
#        # from the top do the backward 
 ##       for i in range(len(self.stack),1):
 ###           input_data = self.stack[i-1].get_input()
##            gradient_data = i.backward(input_data,gradient_data)
            

        
class CrossEntropy(Object):
    # cross entropy
    def __init__(self):
        # init
    def __call__(self,input_data):
        return forward(input_data)
        
    def forward(self,input_data):
        # -log(exp(x[i])/(sum(exp(x[j])) ))
        ee = np.exp(input_data) 
        sf = ee / np.sum(ee)
        return np.sum(-np.log(input_data * sf))
        
    def backward(self,input_data,target_data):
        # - (x[i] - log(\sum(exp(x[j])))
        #  x[i] -1
        # batchsize * 10, 
        idx_of_target = np.equal(input_data,target);
        return input_data - idx_of_target.dtype('uint8')
    
class Optimizer(Object):
    
    def __init__(self,parameters):
        self.alpha = parameters['lr']
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']
        self.stack = []
        self.t = 0
      
        
    def get_stack(self):
        for obj in gc.get_objects():
            if isinstance(obj, Layer):
                self.stack.append(obj);
        #sort the stack by id.
        self.stack.sort(key=lambda x:x.id)
    
    def __call__(self,paramters):
        self.alpha = parameters['lr']
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']
        if not self.stack:
            self.get_stack()
            for i in self.stack:
                
        # the first time 
            
        # optimizer    

if name == '__main__':
    
    # some hyper parameters.
    iteration = 120000
    batch_size = 128
    
    parameters = {
                lr:0.001,
                beta1:0.9,
                beta2:0.999,
            }
     
    # main for loops
    
    _iter = data_loader(batch_size)
    
    
    network = Network()
   
    optimizer = Optimizer(paramters)

    
    for i in range(iteration):
        
        input_image, label = _iter.next()
        
        # forward the netwrok;
        xlabel = Network(input_image)
        
        # backward network;
        loss = CrossEntropy(xlabel,label)
        d_loss_network = CrossEntropy.backward(xlabel,label)
        
        Network.backward(input_image,d_loss_network)
        
        # optimizer parameters
        optimizer(parameters)

        
    # -*- coding: utf-8 -*-

