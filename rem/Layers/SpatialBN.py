import numpy as np
from rem.Layers.Layer import Layer
from rem.Utils.utils import im2col,col2im

class SpatialBN(Layer):
    def __init__(self,channel,**kwags):
        super(SpatialBN,self).__init__(**kwags)
        #https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        self.type = 'sbn'
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