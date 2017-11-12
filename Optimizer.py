import gc
from Layers import Layer
import numpy as np
import math

class Optimizer(object):
    
    def __init__(self,parameters):
        self.lr = parameters['lr']
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']
        self.eps = 1e-8
        self.stack = []
        self.t = 0
           
    def init_stack(self):
        for obj in gc.get_objects():
            if isinstance(obj, Layer):
                self.stack.append(obj);
        #sort the stack by id.
        self.stack.sort(key=lambda x:x.callid)
    
    def __call__(self,parameters):
        self.lr = parameters['lr']
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']
        if not self.stack:
            self.init_stack()
        for layer in self.stack:
            if layer.type == 'conv' or layer.type == 'bn' or layer.type == 'linear':
                
                if layer.m.shape:
                    layer.m = np.zeros(layer.get_weights_grad().shape)
                    layer.v = np.zeros(layer.get_weights_grad().shape)
                    layer.denom = np.zeros(layer.get_weights_grad().shape)
                    
                    layer.mb = np.zeros(layer.get_bias_grad().shape)
                    layer.vb = np.zeros(layer.get_bias_grad().shape)
                    layer.denomb = np.zeros(layer.get_bias_grad().shape)
                
                self.t = self.t + 1
                
                w = layer.get_weights()
                b = layer.get_bias()
                
                
                layer.m = layer.m*self.beta1 + (1-self.beta1)*layer.get_weights_grad()
                layer.v = layer.v*self.beta2 + (1-self.beta2)*layer.get_weights_grad()*layer.get_weights_grad()
                layer.denom = np.sqrt(layer.v) + self.eps
                
                layer.mb = layer.mb*self.beta1 + (1-self.beta1)*layer.get_bias_grad()
                layer.vb = layer.vb*self.beta2 + (1-self.beta2)*layer.get_bias_grad()*layer.get_bias_grad()
                layer.denomb = np.sqrt(layer.vb) + self.eps
                
                bc1 = 1 - self.beta1**self.t
                bc2 = 1 - self.beta2**self.t
                
                ss = - self.lr * math.sqrt(bc2)/bc1 
               
                w = w + layer.m/layer.denom*ss 
                b = b + layer.mb/layer.denomb*ss
                
                layer.set_weights(w)
                layer.set_bias(b)