import numpy as np
from rem.Layers.Layer import Layer
from rem.Utils.utils import im2col,col2im

class Linear(Layer):
    def __init__(self,input_channel,output_channel,**kwags):
        super(Linear,self).__init__(**kwags)
        self.type = 'linear'
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