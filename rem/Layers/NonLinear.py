import numpy as np
from rem.Layers.Layer import Layer
from rem.Utils.utils import im2col,col2im

class NonLinear(Layer):
    def __init__(self,subtype,**kwags):
        super(NonLinear,self).__init__(**kwags)
        self.type =subtype

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