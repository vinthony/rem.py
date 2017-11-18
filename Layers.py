# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:17:08 2017

@author: shado
"""

import time
import numpy as np
import json
from utils import im2col,col2im

class Layer(object):
    counter = 0
    register_id = 0
    def __init__(self,callid=None):
        self.type = 'Layer'
        self.callid = callid
        self.register_id = None
        
        self.m = np.array([])
        self.v = np.array([])
        self.mb = np.array([])
        self.vb = np.array([])
        
    def __call__(self,data):
        if self.callid == None:
            self.callid = Layer.counter
            Layer.counter = Layer.counter + 1
        self.input = data
        return self.forward(data)
       
    def register(self):
        if self.register_id == None:
            self.register_id = Layer.register_id
            Layer.register_id = Layer.register_id + 1

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
        return self.output
        
    def backward(self, input_data, grad_from_back):
        # backward pass
        return self.grad_input
    
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
    def __init__(self,input_channel,output_channel,kernel,stride,padding,**kwags):
        super(Conv2d,self).__init__(**kwags)
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
        self.idx_x = None
        self.idx_y = None

        # can be improved by remembering the idx of im2col and col2im
        
    def forward(self, input_data):
        # bs x ch x w x h
        #t = time.time()
        bs,ch,h,w = input_data.shape
        
        self.input = input_data
        self.output_x = (w + 2*self.padding[0] - self.kernel[0] )//self.stride[0] + 1;
        self.output_y = (h + 2*self.padding[1] - self.kernel[1] )//self.stride[1] + 1;
        
        self.output = np.zeros( (bs,self.output_channel,self.output_y,self.output_x) ) 

        # (bs,ic*k*k,oy*ox)    
        self.imcol_all = im2col(input_data,self.kernel,self.stride,self.padding)
        
        #(oc,1)
        bs_bais = np.reshape(self.bias,[self.output_channel,1])
        
        # (oc , ic*k*k) 
        bs_weight = np.reshape(self.weight,[self.output_channel,-1]);
        
        # (ic*k*k,oy*ox*bs)   
        reshaped_imcol = np.reshape(np.transpose(self.imcol_all,(2,1,0)),(self.input_channel*self.kernel[0]*self.kernel[1],-1))

        #(oc,oy*ox*bs) + (oc,1)
        M = np.dot(bs_weight,reshaped_imcol) + bs_bais ;
        
        self.output = np.transpose(np.reshape(M,(self.output_channel,self.output_y,self.output_x,bs)),(3,0,1,2))

        return self.output
    
    def backward(self, input_data, grad_from_back):
        t = time.time()
        bs,ch,h,w = input_data.shape
        bs,oc,oh,ow = grad_from_back.shape
       
        self.weight_grad.fill(0.0)
        self.grad_input = np.zeros(input_data.shape)
      
        if grad_from_back.ndim < 4: # better to throw a warning here.
            grad_from_back = np.reshape(grad_from_back, (bs,self.output_channel,self.output_y,self.output_x) )

        # (bs,oc,iy,ix)
        self.bias_grad = np.sum(grad_from_back,axis=(0,2,3))
        
        #( x*y, ic*k*k,bs)
        trans_imcol = np.reshape(np.transpose(self.imcol_all,(2,1,0)),(self.output_x*self.output_y,self.input_channel*self.kernel[0]*self.kernel[1]*bs))
        # (bs, oc , oy*ox)
        
        #bs,oc,ic*k*k,bs
        self.weight_grad = np.mean(np.reshape(np.dot(np.reshape(grad_from_back,(bs*oc,oh*ow)),trans_imcol),(bs,oc,ch,self.kernel[0],self.kernel[1],bs)),axis=(0,5))
        
        # ic * k * k x oc, oc x ow*oh*bs
        d_col = np.dot(np.reshape(self.weight,(oc,-1)).T,np.reshape(np.transpose(grad_from_back,(1,2,3,0)),(oc,-1)))
        d_colx = np.transpose(np.reshape(d_col,(self.input_channel*self.kernel[0]*self.kernel[1],self.output_x*self.output_y,bs)),(2,0,1))
        # reshape the d_col to image. d_col[bs,ic*k*k,oh*ow]
        self.grad_input = col2im(d_colx,input_data.shape,self.kernel,self.stride,self.padding)

        return self.grad_input


class SpatialBN(Layer):
    def __init__(self,channel,**kwags):
        super(SpatialBN,self).__init__(**kwags)
        #https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        self.type = 'bn'
        self.channel = channel
        self.eps = 1e-5
        self.weight = np.ones(channel);
        self.bias = np.zeros(channel);
        self.real_mean = np.array([])
        self.real_var = np.array([])
        self.isTrain = True

    def forward(self,input_data):

        bs,c,h,w = input_data.shape
        self.input = input_data
        # update real mean and real var;
        if self.isTrain:
            self.mu = np.mean(input_data,axis=0,keepdims=True)
            self.var = np.var(input_data,axis=0,keepdims=True)
            self.sqrt_v = np.sqrt(self.var + self.eps) ;
            #normalize [bs,ch,h,w]
            self.x_hat = (input_data - self.mu ) / self.sqrt_v 
            # bs x ch x h x w
            self.output = self.x_hat * np.reshape(self.weight,(1,self.channel,1,1)) + np.reshape(self.bias,(1,self.channel,1,1))

            if self.real_mean.shape == np.array([]).shape:
                self.real_mean = np.zeros(self.mu.shape)
                self.real_var = np.ones(self.var.shape)
            else:
                self.real_mean = self.real_mean * 0.9 + self.mu * 0.1
                self.real_var = self.real_var * 0.9 +  self.var * 0.1
        else:
            self.test_sqrt_var = np.sqrt(self.real_var + self.eps) 
            #normalize [bs,ch,h,w]
            self.x_hat = (input_data - self.real_mean ) / self.test_sqrt_var
            # bs x ch x h x w
            self.output = self.x_hat * np.reshape(self.weight,(1,self.channel,1,1)) + np.reshape(self.bias,(1,self.channel,1,1))
        
        return self.output

    def backward(self,input_data,grad_from_back):
        
        bs,c,h,w =  self.input.shape
        if grad_from_back.ndim < 4: # better to throw a warning here.
            grad_from_back = np.reshape(grad_from_back, self.input.shape )
        
        self.bias_grad = np.sum(grad_from_back,axis=(0,2,3)) 
        self.weight_grad = np.sum(self.x_hat*grad_from_back,axis=(0,2,3))
        X_mu = self.input - self.mu
        
        d_hat = np.reshape(self.weight,(1,self.channel,1,1)) * grad_from_back
        
        d_mu = 1. / self.sqrt_v * d_hat
        
        d_norm = X_mu * d_hat
        
        dvar_2 = np.sum(d_norm * -0.5 * (1. /self.sqrt_v)**3,axis=0,keepdims=True)

        d_var = np.sum(d_norm * -1. / self.sqrt_v,axis=0,keepdims=True) + dvar_2 * np.mean(-2.*X_mu,axis=0,keepdims=True)
        
        self.grad_input = d_hat * 1/self.sqrt_v + dvar_2 * 2 * X_mu/bs + d_var/bs 

        return self.grad_input

    def train(self):
        self.isTrain = True

    def evaluate(self):
        self.isTrain = False
   
class BN(Layer):
    def __init__(self,channel,**kwags):
        #https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        super(BN,self).__init__(**kwags)
        self.type = 'bn'
        self.channel = channel
        self.eps = 1e-9
        self.weight = np.random.uniform(0,1,size=(1,channel));
        self.bias = np.zeros((1,channel));
        self.real_mean = np.array([])
        self.real_var = np.array([])
        self.isTrain = True

    def forward(self,input_data):

        bs,c = input_data.shape

        self.input = input_data
        
        if self.isTrain:
            # 1xc
            self.mu = np.mean(input_data,axis=0)
            # 1xc
            self.var = np.var(input_data,axis=0)
            # 
            self.sqrt_v = np.sqrt(self.var + self.eps);
            #normalize [bs,ch]
            self.x_hat = (input_data - self.mu ) / self.sqrt_v 
            # bs x ch 
            self.output = self.x_hat * self.weight + self.bias

            if self.real_mean.shape == np.array([]).shape:
                self.real_mean = np.zeros(self.mu.shape)
                self.real_var = np.ones(self.var.shape)
            else:
                self.real_mean = self.real_mean * 0.9 + self.mu * 0.1
                self.real_var = self.real_var * 0.9 +  self.var * 0.1
        else:
            self.test_sqrt_var = np.sqrt(self.real_var+ self.eps) ;
            #normalize [bs,ch 
            self.x_hat = (input_data - self.real_mean ) / self.test_sqrt_var
            # bs x ch 
            self.output = self.x_hat *  self.weight  +  self.bias 
        
        return self.output

    def backward(self,input_data,grad_from_back):
        bs,c = self.input.shape

        self.bias_grad = np.sum(grad_from_back,axis=0)
        
        self.weight_grad = np.sum(self.x_hat*grad_from_back,axis=0)
        
        X_mu = self.input - self.mu

        std_inv = 1. / np.sqrt(self.var)
        
        d_hat = self.weight * grad_from_back
        
        d_mu = std_inv * d_hat
        
        d_norm = X_mu * d_hat
        
        dvar_2 = np.sum(d_norm * -0.5 * ((1. /self.sqrt_v)**3),axis=0)

        d_var = np.sum(d_norm * -std_inv,axis=0) + dvar_2 * np.mean(-2.*X_mu,axis=0)
        
        self.grad_input = d_hat * std_inv + dvar_2 * 2 * X_mu/bs + d_var/bs 
        
        return self.grad_input

    def train(self):
        self.isTrain = True

    def evaluate(self):
        self.isTrain = False


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
         
        #bs x ic x k*k x oy*ox
        self.output = np.zeros( (bs,self.output_channel,self.output_y,self.output_x) )     
        self.imcol_all = np.reshape(im2col(input_data,self.kernel,self.stride,self.padding),(bs,ch,self.kernel[0]*self.kernel[1],self.output_y,self.output_x))
        
        # bs,op,
        self.output = np.max(self.imcol_all,2)
        self.max_idx = np.argmax(self.imcol_all,2)
        
        return self.output
    
    def backward(self,grad_from_back):
        bs,oc,h,w = grad_from_back.shape
        #(bs,self.input_channel*self.kernel[0]*self.kernel[1],oh*ow)
        #bs ic h*w
        d_col = np.zeros((bs,ic*self.kernel[0]*self.kernel[1],h*w))
        d_col[self.max_idx] = grad_from_back 
        self.grad_input=col2im(d_col,self.input.shape,self.kernel,self.stride,self.padding)
        return self.grad_input

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
        self.output = np.mean(self.imcol_all,2);
        
        return self.output
    
    def backward(self,input_data,grad_from_back):
        bs,oc,h,w = grad_from_back.shape
        #bs ic h*w
        d_col = np.reshape(np.repeat(grad_from_back,(1,self.kernel[0]*self.kernel[1],1,1)),(bs,ic*self.kernel[0]*self.kernel[1],h*w))
        self.grad_input=col2im(d_col,self.input.shape,self.kernel,self.stride,self.padding)
        return self.grad_input
        

class Dropout(Layer):
    def __init__(self,prob,**kwags):
        super(Dropout,self).__init__(**kwags)
        self.type = 'dropout'
        self.prob = prob or 0.5
        self.isTrain = True
    
    def forward(self,input_data):
        self.input = input_data
        if self.isTrain == True: 
            self.saved = 1.0 - self.prob 
            self.sample = np.random.binomial(n=1,p=self.saved,size=input_data.shape)
            input_data = input_data * self.sample
            self.output = input_data / self.saved
        else:
            self.output = self.input
        return self.output
    
    def backward(self,input_data,grad_from_back):

        grad_from_back = grad_from_back.reshape(input_data.shape)

        self.grad_input = grad_from_back / self.saved * self.sample
        
        return self.grad_input

    def train(self):
        self.isTrain = True

    def evaluate(self):
        self.isTrain = False
    

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
        
        # y = W * x + b     bs x ic \times ic x oc + oc
        self.output = np.dot(input_data,self.weight) + np.reshape(self.bias,[1,-1])
           
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
        # [oc x ic]      

        self.bias_grad = np.sum(grad_from_back,axis=0)
        
        # input[bsxocxic],output[bsxocxic],
        self.weight_grad = np.sum(np.reshape(grad_from_back,(bs,1,-1)) * np.reshape(input_data,(bs,-1,1)),axis=0)

        # bs x oc, oc x ic = bs x ic
        self.grad_input = np.dot(grad_from_back,self.weight.T) 

        return self.grad_input



if __name__ == '__main__':
    # c = Conv2d(1,3,[3,3],[1,1],[1,1])
    # c.weight = np.reshape(np.repeat(np.array([[[1,1,1],[1,1,1],[1,1,1]]]),3,axis=0),(3,1,3,3))

    # print(np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12]]]]))
    # o = c(np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12]]]]))

    # print(o)


    b = BN(4)
    o = b(np.array([[1,2,3,4],[5,6,7,8]]))
    c = b.backward(np.array([[1,2,3,4],[5,6,7,8]]),np.array([[1,1,1,1],[0,0,0,0]]))
    print(c)