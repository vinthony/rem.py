import json
import math
from Layers import Conv2d,NonLinear,BN,Linear,Layer,Maxpool,Averagepool
import numpy as np
import gc
import copy

class Model(object):
    def __init__(self):
        self.stack = []
        self.train_flag = True

    def train(self):
        self.train_flag = True
        if not self.stack:
            return
        for i in range(len(self.stack)):
            if self.stack[i].type == 'dropout':
                self.stack[i].train()

    def evaluate(self):
        self.train_flag = False
        for i in self.stack:
            if i.type == 'dropout':
                i.evaluate()

    def getStatus(self):
        return self.train_flag

    def __repr__(self):
        re = dict()
        for k in self.__dict__.keys():
            if hasattr(self,k):
                re[k] = repr(getattr(self,k))
        return json.dumps(re)
    
    def save(self,file):
        re = dict()
        for k in self.__dict__.keys():
            if hasattr(self,k):
                re[k] = repr(getattr(self,k))   
        json.dump(re,file)
        
    def load(self,file):
        # load the network from json.
        with open(file) as f:
            d = json.load(f)
            
        network = Model()
        for k,v in d.items():
            if k == 'stack':
                # build network from stack
                # sort the stack 
                for k2 in json.loads(v):
                    if k2['type'] == 'linear':
                        l = Linear(k2['input_channel'],k2['output_channel'],callid=k2['callid'])
                        l.set_weights(np.asarray(k2['weight']))
                        l.set_bias(np.asarray(k2['bias']))
                    if k2['type'] == 'relu':
                        l = NonLinear('relu',callid=k2['callid'])
                    network.stack.append(l)          
        return network
        
    def forward(self,input_data):
        stack = self.foward_stack();
        for i in stack:
            input_data = i.forward(input_data)
        return input_data
    
    def jsonlize(self):
        return self.__repr__()

    def __call__(self,input_data):
        output = self.forward(input_data)
        return output
    
    def init_stack(self):
        for obj in gc.get_objects():
            if isinstance(obj, Layer):
                obj.register() # register to stack
                self.stack.append(obj);
               
    def getParametersOfModel(self):
        self.model_size = 0
        if not self.stack:
            self.init_stack()
        for i in self.stack:
            if i.get_name() == 'conv2d' or i.get_name() == 'linear' or i.get_name() == 'bn':
                self.model_size = self.model_size + i.get_weights().size + i.get_bias().size
                
        return self.model_size
                
    def foward_stack(self):
        if not self.stack:
            self.init_stack()
        return self.stack
    

    def init(self,init_type):
        if not self.stack:
            self.init_stack()
        # just normal distribution.
        if init_type == 'normal':
            for i in self.stack:
                if i.get_name() == 'conv2d':
                    i.set_weights(np.random.normal(0,0.02,i.get_weights().shape))
                    i.set_bias(np.random.normal(0,0.02,i.get_bias().shape))
                if i.get_name() == 'bn':
                    i.set_weights(np.random.normal(1,0.02,i.get_weights().shape))
                    i.set_bias(np.random.normal(0,0.02,i.get_bias().shape))
                if i.get_name() == 'linear':
                    i.set_weights(np.random.normal(0,0.02,i.get_weights().shape))
                    i.set_bias(np.random.normal(0,0.02,i.get_bias().shape))
        if init_type == 'xavier':
              for i in self.stack:
                if i.get_name() == 'conv2d':
                    i.set_weights(np.random.uniform(-math.sqrt(6/(i.input_channel+i.output_channel)),math.sqrt(6/(i.input_channel+i.output_channel)),i.get_weights().shape))
                    i.set_bias(np.random.uniform(-math.sqrt(6/(i.input_channel+i.output_channel)),math.sqrt(6/(i.input_channel+i.output_channel)),i.get_bias().shape))
                if i.get_name() == 'bn':
                    i.set_weights(np.random.normal(1,0.02,i.get_weights().shape))
                    i.set_bias(np.random.normal(0,0.02,i.get_bias().shape))
                if i.get_name() == 'linear':
                    i.set_weights(np.random.uniform(-math.sqrt(6/(i.input_channel+i.output_channel)),math.sqrt(6/(i.input_channel+i.output_channel)),i.get_weights().shape))
                    i.set_bias(np.random.uniform(-math.sqrt(6/(i.input_channel+i.output_channel)),math.sqrt(6/(i.input_channel+i.output_channel)),i.get_bias().shape))   
        if init_type == 'kaiming':
              for i in self.stack:
                if i.get_name() == 'conv2d':
                    i.set_weights(np.random.normal(0,math.sqrt(1/i.input_channel),i.get_weights().shape))
                    i.set_bias(np.random.normal(0,math.sqrt(1/i.input_channel),i.get_bias().shape))
                if i.get_name() == 'bn':
                    i.set_weights(np.random.normal(1,0.02,i.get_weights().shape))
                    i.set_bias(np.random.normal(0,0.02,i.get_bias().shape))
                if i.get_name() == 'linear':
                    i.set_weights(np.random.normal(0,math.sqrt(1/i.input_channel),i.get_weights().shape))
                    i.set_bias(np.random.normal(0,math.sqrt(1/i.input_channel),i.get_bias().shape))

    def backward_stack(self):
        if not self.stack:
            self.init_stack()
        self.stack.sort(key=lambda x:x.callid)
        self.stack = self.stack[::-1]
        return self.stack
   
    def backward(self,input_data,grad_from_output):
        stack = self.backward_stack();
        for i in stack:
            grad_from_output = i.backward(i.input,grad_from_output)
        return grad_from_output
    
    

        


