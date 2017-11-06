# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:17:08 2017

@author: shado
"""

import time
import numpy as np


def im2col(imgray,k,stride,padding):
    r = k[0]//2;
    c = k[1]//2;
    
    sr = stride[0];
    sc = stride[1];
    
    pr = padding[0];
    pc = padding[1];

    h,w = imgray.shape
    
    re = np.zeros(r*c,(h-1)*stride * (w-1)*stride);
    for y in range(h):
        for x in range(w):
            # if border, padding
            if y == 0 or x == 0 or y == h-1 or x == w-1:
                re[] = 
            else:
                re[:,w*(x-1)+h] = imgray[y-r:y+r,x-r:x+c]
    
    return re;

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
    def get_weights(self):
        return self.weight
    
    def get_bais(self):
        return self.bais

    def set_weights(self,weight):
        self.weight = weight

    def set_bais(self,bais):
        self.bais = bais


class NonLinear(Object):
    
    def __init__(self,subtype):
        self.type =subtype

    def __call__(self,data):
        if self.id == None:
            self.id = time.time()
            
        return self.forward(data)

    def forward(self, input_data):
        if self.type.lower() == 'relu':
            self.output = np.maximum(input_data,np.zeros(input_data.shape)
       
        
    def backward(self, input_data, grad_from_back):
        
        if self.type.lower() == 'relu':
            self.grad = grad_from_back * np.greater(input_data,0).astype('byte')
        return self.grad
    
class Conv2d(Layer):
    def __init__(self,input_channel,output_channel,kernel,stride=1,padding):
        super.__init__(self)
        self.type = 'conv2d'
        self.input_channel = input_channel
        self.output_channel  = output_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = np.zeros((output_channel,kernel,kernel))
        self.bais  = np.zeros(output_channel)
        self.adamM = 0
        self.adamV = 0
        
    def __call__(self,data):
        if self.id == None:
            self.id = time.time()
            
        return self.forward(data)
        
    def forward(self, input_data):
        # bs x ch x w x h
        bs,ch,w,h = input_data.shape
        
        self.output_x = (w-2*self.padding[0] + self.kernel[0] )//self.stride[0] + 1;
        self.output_y = (h-2*self.padding[1] + self.kernel[1] )//self.stride[1] + 1;
        
        self.output = np.zeros(bs,self.output_channel,self.output_y,self.output_x)
        
        for i in range(bs):
            imcol_all = np.zeros(ch*self.kernel[0]*self.kernel[1],self.output_x*self.output_y)
            for j in range(ch):
                imcol_all[j:j+self.kernel[0]*self.kernel[1],-1] = im2col(input_data[i,j,:,:],self.kernel,self.stride,self.padding)
            imkol_all = np.repeat( np.resize(self.weight,(self.output_channel,(self.kernel[0]*self.kernel[1])),ch, axis=1)
            im = np.reshape(np.dot(imkol_all,imcol_all),(self.output_channel,self.output_y,self.output_x)); # [oc , ch*kk ] x [ch*kk,ox*oy]
            self.output[i] = im;
        return self.output
    
    def backward(self, input_data, grad_from_back):
        pass

   

   



class Linear(Layer):
    def __init__(self,input_channel,output_channel,kernel,stride=1,padding):
        super.__init__(self)
        self.type = 'linear'
        self.input_channel = input_channel
        self.output_channel  = output_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = np.zeros(input_channel)
        self.weight_grad = np.zeros(input_channel)
        self.bais  = np.zeros(input_channel)
        self.bais_grad = np.zeros(input_channel)
        self.adamM = 0
        self.adamV = 0
        
    def __call__(self,data):
        if self.id == None:
            self.id = time.time()
            
        return self.forward(data)
        
    def forward(self, input_data):
        # bs x c
        
        bs,c = input_data.shape
        
        self.output = np.zeros((bs,self.output_channel))
        
        for i in range(bs):
            _m = np.repeat(self.input_data[i],self.output_channel,axis=0)# OC x IC
            self.output[i] = np.dot(_m,self.weight) + self.bais # [ic] = [ocxoc]
           
        return self.output
        
        
    def backward(self, input_data, grad_from_back):
        pass
    