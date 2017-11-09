# -*- coding: utf-8 -*-
"""
Spyder Editor 2017年11月2日23:41:00

This is a temporary script file.
"""

import json
import numpy as np
import struct
import gc


from Layers import Conv2d,NonLinear,BN,Linear,Layer,Maxpool,Averagepool

def data_loader(image,label,batch_size):
    # generater an image from datasert
    with open(label,'rb') as lb:
        # file header [0,9]
        m,n = struct.unpack('>II',lb.read(8))
        labels = np.fromfile(lb, dtype=np.uint8)

    with open(image,'rb') as im:
        m,n,r,c = struct.unpack('>IIII',im.read(16))
        images = np.fromfile(im,dtype=np.uint8).reshape(len(labels),28,28)

    length = len(labels)

    while True:
        idxs = np.arange(length)
        np.random.shuffle(idxs)

        for batch_idx in range(0,length,batch_size):

            batch_label = labels[batch_idx:batch_idx+batch_size]
            batch_image = images[batch_idx:batch_idx+batch_size]
            
            yield batch_image,batch_label
    
    
class Network(object):
    def __init__(self):
        self.linear1 = Linear(28*28,14*14);
        self.linear2 = Linear(14*14,7*7);
        self.linear3 = Linear(7*7,10);
        self.relu = NonLinear(subtype='relu')
        self.stack = []

    def __call__(self,input_data):
        return self.forward(input_data)
    
    def init_stack(self):
        for obj in gc.get_objects():
            if isinstance(obj, Layer):
                self.stack.append(obj);
        #sort the stack by id.
        self.stack.sort(key=lambda x:x.callid)
        self.stack = self.stack[::-1]

    def init(self):
        if not self.stack:
            self.init_stack()
        # just normal distribution.
        for i in self.stack:
            if i.get_name() == 'conv2d':
                i.set_weights(np.random.normal(0,0.02,i.get_weight().shape))
                i.set_bais(np.random.normal(0,0.02,i.get_bais().shape))
            if i.get_name() == 'bn':
                i.set_weights(np.random.normal(1,0.02,i.get_weight().shape))
                i.set_bais(np.random.normal(1,0.02,i.get_bais().shape))
            if i.get_name() == 'linear':
                i.set_weights(np.random.normal(0,0.02,i.get_weight().shape))
                i.set_bais(np.random.normal(0,0.02,i.get_bais().shape))

    def get_stack(self):
        return self.stack

        
    def forward(self,input_data):
        
        op = self.linear3(
                    self.relu(self.linear2(
                        self.relu(self.linear1(input_data))
                        )
                    )
                )
        
        return op
    
    def backward(self,input_data,grad_from_output):
        if not self.stack:
            self.get_stack();
        for i in self.stack:
            grad_from_output = i.backward(i.input,grad_from_output)
        return grad_from_output
        

        
class CrossEntropy(object):
    # cross entropy
    def __init__(self):
        self.eps = 1e-9
        pass
    def __call__(self,input_data,target_data):
        return self.forward(input_data,target_data)
        
    def forward(self,input_data,target_data):
        # bs x 10
        ee = np.exp(input_data) 
        sf = ee / np.sum(ee,axis=1).reshape((-1,1))

        self.softmax = sf;
        
        self.output = np.mean(np.sum(-target_data * np.log(sf+self.eps),axis=1))
        
        return self.output
        
    def backward(self,input_data,target_data):
        # \sum( - t * log(exp(i)/ \sum(exp(i)) )) 
        # batchsize * 10, 

        return self.softmax - target_data
    
def getMatrixOfClass(target_data,labels=10):
    bs, = target_data.shape
    mx = np.zeros((bs,10))
    idx = np.arange(bs)
    mx[idx[:],target_data[:]] = 1;
    return mx

class Optimizer:
    
    def __init__(self,parameters):
        self.lr = parameters['lr']
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']
        self.stack = []
        self.t = 0
      
        
    def init_stack(self):
        for obj in gc.get_objects():
            if isinstance(obj, Layer):
                self.stack.append(obj);
        #sort the stack by id.
        self.stack.sort(key=lambda x:x.callid)
    
    def __call__(self,paramters):
        self.lr = parameters['lr']
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']
        if not self.stack:
            self.init_stack()
        for layer in self.stack:
            if layer.type == 'conv' or layer.type == 'bn' or layer.type == 'linear':
                w = layer.get_weights()
                b = layer.get_bais()
                # update the parameters
                w = w - self.lr * layer.get_weights_grad()
                b = b - self.lr * layer.get_bais_grad()
                
                layer.set_weights(w)
                layer.set_bais(b)

if __name__ == '__main__':

    # some hyper parameters.
    iteration = 120000
    batch_size = 128
    
    traing_samples = 'train-images.idx3-ubyte' 
    traing_labels = 'train-labels.idx1-ubyte'

    parameters = {
                "lr":0.001,
                "beta1":0.9,
                "beta2":0.999,
            }
     
    # main for loops
    
    _iter = data_loader(traing_samples,traing_labels,batch_size)
    
    network = Network()

    network.init()
   
    optimizer = Optimizer(parameters)

    criterion = CrossEntropy()
    
    for i in range(iteration):
        
        input_image, label = _iter.next()

        label = getMatrixOfClass(label)

        # forward the ;
        xlabel = network(np.reshape(input_image,(-1,28*28)))
        # backward network;
        loss = criterion(xlabel,label)

        print(loss)

        d_loss_network = criterion.backward(xlabel,label)
        
        network.backward(input_image,d_loss_network)
        
        # optimizer parameters
        optimizer(parameters)

        
