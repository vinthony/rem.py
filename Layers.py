# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:17:08 2017

@author: shado
"""

import time

def padding_to_same():


def im2col(imgray,k):
    r = k[0];
    c = k[1];

    h,w = imgray.shape

    for y in range(h):
        for x in range(w):
            imgray[y:y+r]


class Layer(Object):
    
    def __init__(self):
        self.type = 'Layer'

    def __call__(self,data):
        if self.id == None:
            self.id = time.time()
            
        return self.forward(data)
    
    def get_name(self):
        return self.type

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
        self.weight = np.zeros(bs,output_channel,kernel,kernel)
        self.bais  = np.zeros(bs,output_channel)
        self.adamM = 0
        self.adamV = 0
        
    def __call__(self,data):
        if self.id == None:
            self.id = time.time()
            
        return self.forward(data)
        
    def forward(self, input_data):
        # bs x ch x w x h
        bs,ch,w,h = input_data.shape
        
        for i in range(bs):
            for j in range(ch):
                input_data[i,j,:,:]


        
    def backward(self, input_data, grad_from_back):
        pass
    
    def update_parameters(self):
        pass

    def get_variables(self):
        return self.weight
    
    def get_bais(self):
        return self.bais

    def set_variables(self,weight):
        self.weight = weight

    def set_bais(self,bais):
        self.bais = bais

   



class Linear(Layer):
    def __init__(self,input_channel,output_channel,kernel,stride=1,padding):
        super.__init__(self)
        self.type = 'conv2d'
        self.input_channel = input_channel
        self.output_channel  = output_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = np.zeros(bs,output_channel,kernel,kernel)
        self.bais  = np.zeros(bs,output_channel)
        self.adamM = 0
        self.adamV = 0
        
    def __call__(self,data):
        if self.id == None:
            self.id = time.time()
            
        return self.forward(data)
        
    def forward(self, input_data):
        # bs x ch x w x h
        
        
    def backward(self, input_data, grad_from_back):
        pass
    
    def update_parameters(self):
        pass