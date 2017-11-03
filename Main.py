# -*- coding: utf-8 -*-
"""
Spyder Editor 2017年11月2日23:41:00

This is a temporary script file.
"""

import json
import numpy as np
import struct


from Layers import Conv2d,NonLinear,BN,Linear

def data_loader(image,label,batch_size):
    # generater an image from datasert
    with open(label,'rb') as lb:
        # file header [0,9]
        m,n = struct.unpack('>II',lb.read(8))
        labels = np.fromfile(lb, dtype=np.uint8)

    with open(image,'rb') as im:
        m,n,r,c = struct.unpack('>IIII',im.read(16))
        images = np.fromfile(lb,dtype=np.uint8).reshape(len(labels),28,28)

    length = len(labels)

    while True:
        idxs = np.arange(labels)
        np.random.shuffle(idxs)

        for batch_idx in range(0,length,batch_size):

            batch_label = labels[batch_idx:batch_idx+batch_size]
            batch_image = images[batch_idx:batch_idx+batch_size]
            
            yield batch_image,batch_label
    
    
class Network(Object):
    def __init__(self,**kwarg):
        self.conv1 = Conv2d(3,64,3,3,1,1,1,1);
        self.bn1 = BN(64);
        self.maxpool = Maxpool(2); # 128x14x14
        self.conv2 = Conv2d(64,64,3,3,1,1,1,1);
        self.bn2 = BN();
        self.averagepool = Averagepool(2); # 128x7x7
        self.relu  = NonLinear(sub_type='relu');
        self.softmax = NonLinear(sub_type='softmax');
        self.linear1 = Linear(49);
        self.linear2 = Linear(10);
        self.is_init = False
        self.stack = []

    def __call__(self,input_data):
        return self.forward(input_data)
    
    def init_stack(self):
        for obj in gc.get_objects():
            if isinstance(obj, Layer):
                self.stack.append(obj);
        #sort the stack by id.
        self.stack.sort(key=lambda x:x.id)

    def init(self):
        if not self.stack:
            self.init_stack()
        # just normal distribution.
        for i in self.stack:
            if i.get_name() == 'conv2d':
                i.set_variables(np.normal.normal(0,0.02,i.get_variables().shape))
                i.set_bais(np.normal.normal(0,0.02,i.get_bais().shape))
            if i.get_name == 'bn':
                i.set_variables(np.normal.normal(1,0.02,i.get_variables().shape))
                i.set_bais(np.normal.normal(1,0.02,i.get_bais().shape))
            if i.get_name == 'linear':
                i.set_variables(np.normal.normal(0,0.02,i.get_variables().shape))
                i.set_bais(np.normal.normal(0,0.02,i.get_bais().shape))

    def get_stack(self):
        return self.stack

        
    def forward(self,input_data):
        
        x1 = self.bn1(self.relu(self.conv1(input_data)))
        
        x2 = self.bn2(self.relu(self.conv2(x1)))
        
        x3 = self.softmax(self.linear1(self.linear2(x2)))
        
        return x3
        

        
class CrossEntropy(Object):
    # cross entropy
    def __init__(self):
        # init
    def __call__(self,input_data):
        return forward(input_data)
        
    def forward(self,input_data):
        # -log(exp(x[i])/(sum(exp(x[j])) ))
        ee = np.exp(input_data) 
        sf = ee / np.sum(ee)
        return np.sum(-np.log(input_data * sf))
        
    def backward(self,input_data,target_data):
        # - (x[i] - log(\sum(exp(x[j])))
        #  x[i] -1
        # batchsize * 10, 
        idx_of_target = np.equal(input_data,target);
        return input_data - idx_of_target.dtype('uint8')
    
class Optimizer(Object):
    
    def __init__(self,parameters):
        self.alpha = parameters['lr']
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']
        self.stack = []
        self.t = 0
      
        
    def get_stack(self):
        for obj in gc.get_objects():
            if isinstance(obj, Layer):
                self.stack.append(obj);
        #sort the stack by id.
        self.stack.sort(key=lambda x:x.id)
    
    def __call__(self,paramters):
        self.alpha = parameters['lr']
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']
        if not self.stack:
            self.get_stack()
            for i in self.stack:
                
        # the first time 
            
        # optimizer    

if name == '__main__':
    
    # some hyper parameters.
    iteration = 120000
    batch_size = 128
    
    traing_samples = 'train-images.idx3-ubyte' 
    traing_labels = 'train-labels.idx1-ubyte'

    parameters = {
                lr:0.001,
                beta1:0.9,
                beta2:0.999,
            }
     
    # main for loops
    
    _iter = data_loader(batch_size)
    
    
    network = Network()
   
    optimizer = Optimizer(paramters)

    
    for i in range(iteration):
        
        input_image, label = _iter.next()
        
        # forward the netwrok;
        xlabel = Network(input_image)
        
        # backward network;
        loss = CrossEntropy(xlabel,label)
        d_loss_network = CrossEntropy.backward(xlabel,label)
        
        Network.backward(input_image,d_loss_network)
        
        # optimizer parameters
        optimizer(parameters)

        
