import numpy as np
from rem.Layers.Layer import Layer
from rem.Utils.utils import im2col,col2im

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