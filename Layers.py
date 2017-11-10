# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:17:08 2017

@author: shado
"""

import time
import numpy as np
import json

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
                # re[] = 
                print('border')
            else:
                re[:,w*(x-1)+h] = imgray[y-r:y+r,x-r:x+c]
    
    return re;

class Layer(object):
    
    def __init__(self,callid=None):
        self.type = 'Layer'
        self.callid = callid

    def __call__(self,data):
        if self.callid == None:
            self.callid = time.time()
            
        return self.forward(data)
    
    def __repr__(self):
        dic  = {}
        for k,v in self.__dict__.items():
            if type(v) == type(np.zeros(1)):
                if k == 'weight' or k == 'bias':
                    v = v.tolist()
                else:
                    continue
            dic[k] = v 
        return json.dumps(dic)


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
    
    def get_bias(self):
        return self.bias

    def set_weights(self,weight):
        self.weight = weight

    def set_bias(self,bias):
        self.bias = bias
    
    def get_weights_grad(self):
        return self.weight_grad
    
    def get_bias_grad(self):
        return self.bias_grad


class NonLinear(object):
    
    def __init__(self,subtype,**kwags):
        super(NonLinear,self).__init__(**kwags)
        self.type =subtype
        self.id = time.time()

    def __call__(self,data):
        if self.id == None:
            self.id = time.time()
        return self.forward(data)

    def forward(self, input_data):
        if self.type.lower() == 'relu':
            self.output = np.maximum(input_data,np.zeros(input_data.shape))
        return self.output

    def backward(self, input_data, grad_from_back):
        if self.type.lower() == 'relu':
            self.grad = grad_from_back * np.greater(input_data,0).astype('byte')
        return self.grad
    
class Conv2d(Layer):
    def __init__(self,input_channel,output_channel,kernel,stride,padding):
        super(Conv2d,self).__init__()
        self.type = 'conv2d'
        self.input_channel = input_channel
        self.output_channel  = output_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = np.zeros((output_channel,kernel[0],kernel[1]))
        self.bias  = np.zeros(output_channel)
        
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
            imkol_all = np.repeat( np.resize(self.weight,(self.output_channel,(self.kernel[0]*self.kernel[1]))),ch, axis=1)
            im = np.reshape(np.dot(imkol_all,imcol_all),(self.output_channel,self.output_y,self.output_x)); # [oc , ch*kk ] x [ch*kk,ox*oy]
            self.output[i] = im;
        return self.output
    
    def backward(self, input_data, grad_from_back):
        pass

   
class BN(Layer):
    def __init__(self,channel):
        self.type = 'bn'
        self.channel = channel
        self.eps = 1e-5
        self.weight = np.zeros(channel);
        self.bias = np.zeros(channel);

    def forward(self,input_data):

        bs,c,w,h = self.input_data.shape

        # mean [batch,ch,w,h]
        self.m = np.mean(input_data,axis=0,keepdims=True)
        # variance
        self.v = np.var(input_data,axis=0,keepdims=True)
        #normalize [bs,ch,w,h]
        self.x_hat = (input_data - np.repeat(self.m,bs,axis=0) ) / (np.repeat(np.sqrt(self.v),bs,axis=0) + self.eps)

        self.output = self.x_hat * self.weight + self.bias

        return self.output

    def backward(self,input_data,grad_from_back):

        self.bias_grad = np.sum()


class Maxpool(Layer):
    def __init__(self):
        self.type = 'maxpool'

class Averagepool(Layer):
    def __init__(self):
        self.type = 'averagepool'


class Linear(Layer):
    def __init__(self,input_channel,output_channel,**kwags):
        super(Linear,self).__init__(**kwags)
        self.type = 'linear'
        self.id = time.time()
        self.input_channel = input_channel
        self.output_channel  = output_channel
        self.weight = np.zeros( (input_channel,output_channel) )
        self.weight_grad = np.zeros( (input_channel,output_channel) )
        self.bias  = np.zeros(output_channel)
        self.bias_grad = np.zeros(output_channel)
        self.grad_input = np.zeros(input_channel)

    def __call__(self,data):
        if self.callid == None:
            self.callid = time.time()
            
        return self.forward(data)
        
    def forward(self, input_data):
        # bs x ic
        bs,c = input_data.shape
        self.input = input_data
        #bs x oc
        self.output = np.zeros((bs,self.output_channel))
        
        for i in range(bs):
            # y = W * x + b     ic \times ic x oc + oc
            self.output[i] = np.dot(input_data[i],self.weight) + self.bias
           
        return self.output
        
        
    def backward(self, input_data, grad_from_back):
        # w*X + b = y
        # dy / db = 1 * g_f_b
        # dy / dw = X * g_f_b
        # dy / dX = w * g_f_b

        # [bs x ic] 
        bs,c = input_data.shape

        # bs x oc  grad_from_back

        self.weight_grad.fill(.0)
        self.bias_grad.fill(.0)
        self.grad_input = np.zeros((bs,c))
        # [oc x ic]      
        for i in range(bs):
            # [ oc * ic ]
            # oc x ic \times  oc x ic
            self.weight_grad += np.transpose(np.multiply(np.repeat(grad_from_back[i].reshape(-1,1),self.input_channel,axis=1), np.repeat(input_data[i].reshape(1,-1),self.output_channel,axis=0)) )
            self.bias_grad += grad_from_back[i]
            self.grad_input[i] =  np.dot(self.weight, grad_from_back[i]) 

        self.weight_grad /= bs
        self.bias_grad /= bs 
        self.grad_input /= bs 

        return self.grad_input






    