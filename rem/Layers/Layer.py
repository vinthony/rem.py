import numpy as np
import json

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