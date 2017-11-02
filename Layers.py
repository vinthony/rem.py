# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:17:08 2017

@author: shado
"""

import time

class Layer(Object):
    
    def __init__(self):
        # 
        pass
    def __call__(self,data):
        if self.id == None:
            self.id = time.time()
            
        return self.forward(data)
    
    def forward(self, input_data):
        # forward pass
        pass
        
    def backward(self, input_data, grad_from_back):
        
        # backward pass
        pass

class Conv2d(Layer):
    def __init__(self,input_channel,output_channel,kernel,stride=1,padding):
        super.__init__(self)
        self.type = 'conv2d'
        self.input_channel = input_channel
        self.output_channel  = output_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        
        self.adamM = 0
        self.adamV = 0
        
    def __call__(self,data):
        if self.id == None:
            self.id = time.time()
            
        return self.forward(data)
        
    def forward(self, input_data):
        pass
        
    def backward(self, input_data, grad_from_back):
        pass
    
    def update_parameters(self):
        pass