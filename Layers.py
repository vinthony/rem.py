# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:17:08 2017

@author: shado
"""

import time
import numpy as np
import json
import uuid

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


from utils import im2col,imremap

class Layer(object):
    counter = 0
    def __init__(self,callid=None):
        self.type = 'Layer'
        self.callid = None
        
        self.m = np.array([])
        self.v = np.array([])
        self.mb = np.array([])
        self.vb = np.array([])
        
    def __call__(self,data):
        if self.callid == None:
            self.callid = Layer.counter
            Layer.counter = Layer.counter + 1
       
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
        return self
        
    def backward(self, input_data, grad_from_back):
        # backward pass
        return self
    
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


class NonLinear(Layer):
    def __init__(self,subtype,**kwags):
        super(NonLinear,self).__init__(**kwags)
        self.type =subtype
        self.id = time.time()

    def forward(self, input_data):
        self.input = input_data
        if self.type.lower() == 'relu':
            self.output = np.maximum(input_data,np.zeros(input_data.shape))
        return self.output

    def backward(self, input_data, grad_from_back):
        
        if self.input.shape != grad_from_back.shape:
            grad_from_back = np.reshape(grad_from_back,input_data.shape)

        if self.type.lower() == 'relu':
            self.grad_input = grad_from_back * np.greater(input_data,0).astype('byte')
        return self.grad_input
    
class Conv2d(Layer):
    def __init__(self,input_channel,output_channel,kernel,stride,padding):
        super(Conv2d,self).__init__()
        self.type = 'conv2d'
        self.input_channel = input_channel
        self.output_channel  = output_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = np.zeros((output_channel,input_channel,kernel[0],kernel[1]))
        self.weight_grad = np.zeros((output_channel,input_channel,kernel[0],kernel[1]))
        self.bias  = np.zeros(output_channel)
        self.bias_grad = np.zeros(output_channel)
        
    def forward(self, input_data):
        # bs x ch x w x h
        #t = time.time()
        bs,ch,h,w = input_data.shape
        
        self.input = input_data
        self.output_x = (w + 2*self.padding[0] - self.kernel[0] )//self.stride[0] + 1;
        self.output_y = (h + 2*self.padding[1] - self.kernel[1] )//self.stride[1] + 1;
        
        self.output = np.zeros( (bs,self.output_channel,self.output_y,self.output_x) )     
        self.imcol_all = im2col(input_data,self.kernel,self.stride,self.padding)
        bs_bais = np.tile(np.reshape(self.bias,[1,self.output_channel,1]),(bs,1,self.output_x*self.output_y));
        # bs x oc x ic*k*k x oy*ox
#        print(np.repeat(np.expand_dims(self.imcol_all,axis=1) ,self.output_channel,axis=1).shape,np.tile(np.reshape(self.weight,[1,self.output_channel,-1,1]),(bs,1,1,self.output_x*self.output_y)).shape,bs_bais.shape)
        self.output = np.sum(np.repeat(np.expand_dims(self.imcol_all,axis=1) ,self.output_channel,axis=1) *  np.tile(np.reshape(self.weight,[1,self.output_channel,-1,1]),(bs,1,1,self.output_x*self.output_y)) , axis=2) + bs_bais;
        
        self.output = np.reshape(self.output,(bs,self.output_channel,self.output_y,self.output_x))
        
        #print('forward',time.time()-t,self.callid)
        return self.output
    
    def backward(self, input_data, grad_from_back):
        t = time.time()
        bs,ch,h,w = input_data.shape
        bs,oc,oh,ow = grad_from_back.shape
        
        #print(input_data.shape,grad_from_back.shape)
        if grad_from_back.ndim < 4:
            grad_from_back = np.reshape(grad_from_back, (bs,self.output_channel,self.output_y,self.output_x) )

        self.bias_grad = np.sum(np.sum(np.sum(grad_from_back,axis=3),axis=2),axis=0)
        self.transposed_weight = np.rot90(self.weight,k=2,axes=(2,3))  # rotate 180 for derivate
        self.grad_input = np.zeros((bs,ch,h,w))
        
        # bs x oc x ich*k*k x oy*ox
        imcolxx = np.repeat(np.expand_dims(self.imcol_all,axis=1) ,self.output_channel,axis=1)
        gradxx = np.tile(np.expand_dims(np.reshape(grad_from_back,(bs,oc,oh*ow)),axis=2),(1,1,ch*self.kernel[0]*self.kernel[1],1))
        
        self.weight_grad = np.reshape(np.sum(np.sum(imcolxx * gradxx,axis=3),axis=0),self.weight.shape);
        
        # bs x oc x oy x ox
        # bs x  oc*k*k x h*w
        self.imcol_grad_all = imremap(grad_from_back,h,w,self.kernel,self.stride,self.padding)     
        
        #print(grad_from_back.shape,self.imcol_grad_all.shape,self.input_channel)
        # ic * oc * k * k 
        extend_transpose_weight = np.tile(np.reshape(self.transposed_weight,[1,self.input_channel,-1,1]),(bs,1,1,h*w))
        
#       
        # bs x ic x oc*k*k x h * w;
        self.grad_input = np.mean(np.repeat(np.expand_dims(self.imcol_grad_all,axis=1) ,self.input_channel,axis=1) * extend_transpose_weight, axis=2)
        
        self.grad_input = np.reshape(self.grad_input,(bs,self.input_channel,h,w))
        
        #print('backward',time.time()-t,self.callid)
        return self.grad_input


class SpatialBN(Layer):
    def __init__(self,channel):
        super(SpatialBN,self).__init__()
        #https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        self.type = 'bn'
        self.channel = channel
        self.eps = 1e-5
        self.weight = np.zeros(channel);
        self.bias = np.zeros(channel);

    def forward(self,input_data):

        bs,c,h,w = self.input_data.shape

        # mean [batch,ch,h,w]
        self.m = np.mean(input_data,axis=0,keepdims=True)
        # variance
        self.v = np.var(input_data,axis=0,keepdims=True)
        
        self.sqrt_v = np.sqrt(self.v) + self.eps;
        #normalize [bs,ch,h,w]
        self.x_hat = (input_data - np.repeat(self.m,bs,axis=0) ) / (np.repeat(self.sqrt_v,bs,axis=0) )
        
        # bs x ch x h x w
        self.output = self.x_hat * np.repeat(self.weight,[bs,1,h,w]) + np.repeat(self.bias,[bs,1,h,w])

        return self.output

    def backward(self,input_data,grad_from_back):
        
        bs,c,h,w = self.input_data.shape

        self.bias_grad = np.sum(np.sum(np.sum(grad_from_back,axis=3),axis=2),axis=0)
        
        self.weight_grad = np.sum(np.sum(np.sum(self.x_hat*grad_from_back,axis=3),axis=2),axis=0)
        
        d_hat = np.repeat(self.weight,[bs,1,h,w]) * grad_from_back
        
        d_sqrt_var = - 1. / (self.sqrt_v**2) * d_hat
        # bs x c x h x w
        dvar = 0.5 * 1 / self.sqrt_v * d_sqrt_var
        
        dsq = np.ones((bs,c,w,h))/bs * dvar;
        
        d_v = 2*self.m * dsq
        
        d_m = d_hat * self.v
        
        d_mv = d_v + d_m
        
        d_u = -1 * np.sum(d_mv,axis=0)
        
        d_x2 = np.ones((bs,c,w,h))/bs * np.repeat(d_u,(bs,1,1,1))
        
        self.grad_input = d_x2 + d_mv
    
        return self.grad_input
   
class BN(Layer):
    def __init__(self,channel):
        #https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        super(BN,self).__init__()
        self.type = 'bn'
        self.channel = channel
        self.eps = 1e-5
        self.weight = np.zeros(channel);
        self.bias = np.zeros(channel);

    def forward(self,input_data):

        bs,c = input_data.shape
        
        self.input = input_data
        
        # mean [batch,ch,h,w]
        self.m = np.mean(input_data,axis=0,keepdims=True)
        # variance
        self.v = np.var(input_data,axis=0,keepdims=True)
        
        self.sqrt_v = np.sqrt(self.v) + self.eps;
        #normalize [bs,ch,h,w]
        self.x_hat = (input_data - np.repeat(self.m,bs,axis=0) ) / (np.repeat(self.sqrt_v,bs,axis=0) )

        # bs x ch x h x w
        self.output = self.x_hat * np.repeat(np.reshape(self.weight,(1,-1)),bs,axis=0) + np.repeat(np.reshape(self.bias,(1,-1)),bs,axis=0)

        return self.output

    def backward(self,input_data,grad_from_back):
        
        bs,c = self.input.shape

        self.bias_grad = np.sum(grad_from_back,axis=0)
        
        self.weight_grad = np.sum(self.x_hat*grad_from_back,axis=0)
        
        d_hat = np.repeat(np.reshape(self.weight,(1,-1)),bs,axis=0) * grad_from_back
        
        d_sqrt_var = - 1. / (self.sqrt_v**2) * d_hat
        # bs x c x h x w
        dvar = 0.5 * 1 / self.sqrt_v * d_sqrt_var
        
        dsq = np.ones((bs,c))/bs * dvar;
        
        d_v = 2*self.m * dsq
        
        d_m = d_hat * self.v
        
        d_mv = d_v + d_m
        
        d_u = -1 * np.sum(d_mv,axis=0)
        
        d_x2 = np.ones((bs,c))/bs * np.repeat(np.reshape(d_u,(1,-1)),bs,axis=0)
        
        self.grad_input = d_x2 + d_mv
    
        return self.grad_input


class Maxpool(Layer):
    def __init__(self,kernel,stride,padding):
        super(Maxpool,self).__init__()
        self.type = 'maxpool'
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        
    def forward(self,input_data):
        bs,ch,h,w = input_data.shape
        
        self.input = input_data
        self.output_x = (w + 2*self.padding[0] - self.kernel[0] )//self.stride[0] + 1;
        self.output_y = (h + 2*self.padding[1] - self.kernel[1] )//self.stride[1] + 1;
         
        #bs x ic*k*k x oy*ox
        self.output = np.zeros( (bs,self.output_channel,self.output_y,self.output_x) )     
        self.imcol_all = np.reshape(im2col(input_data,self.kernel,self.stride,self.padding),(bs,ch,self.kernel[0]*self.kernel[1],self.output_x*self.output_y))
        
        # bs,op,
        self.output = np.max(self.imcol_all,3)
        
        return self.output
    
    def backward(self):
        pass
    

class Averagepool(Layer):
    def __init__(self,kernel,stride,padding):
        super(Averagepool,self).__init__()
        self.type = 'avepool'
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        
    def forward(self,input_data):
        bs,ch,h,w = input_data.shape
        
        self.input = input_data
        self.output_x = (w + 2*self.padding[0] - self.kernel[0] )//self.stride[0] + 1;
        self.output_y = (h + 2*self.padding[1] - self.kernel[1] )//self.stride[1] + 1;
         
        #bs x ic*k*k x oy*ox
        self.output = np.zeros( (bs,self.output_channel,self.output_y,self.output_x) )     
        self.imcol_all = np.reshape(im2col(input_data,self.kernel,self.stride,self.padding),(bs,ch,self.kernel[0]*self.kernel[1],self.output_x*self.output_y))
        
        # bs,op,
        self.output = np.mean(self.imcol_all,3);
        
        return self.output
    
    def backward(self,input_data,grad_from_back):
       
       pass

class Dropout(Layer):
    def __init__(self,prob,**kwags):
        super(Dropout,self).__init__(**kwags)
        self.prob = prob
    
    def forward(self,input_data):
        self.input = input_data
        self.saved = 1.0 - self.prob
        self.sample = np.random.binomial(n=1,p=self.saved,size=input_data.shape)
        
        input_data = input_data * self.sample
        
        self.output = input_data / self.saved
        
        return self.output
    
    def backward(self,input_data,grad_from_back):
        
        self.grad_input = grad_from_back / self.saved * self.sample
        
        return self.grad_input
    

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

    def forward(self, input_data):
        # bs x ic
        bs = input_data.shape[0]

        input_data = np.reshape(input_data,(bs,-1))

        ch = self.input_channel

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



if __name__ == '__main__':
    c = Conv2d(1,3,[3,3],[1,1],[1,1])
    c.weight = np.reshape(np.repeat(np.array([[[1,1,1],[1,1,1],[1,1,1]]]),3,axis=0),(3,1,3,3))

    print(np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12]]]]))
    o = c(np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12]]]]))

    print(o)