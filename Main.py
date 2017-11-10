# -*- coding: utf-8 -*-
"""
Spyder Editor 2017年11月2日23:41:00

This is a temporary script file.
"""

import numpy as np

from Models import Model

from Optimizer import Optimizer
from Criterions import CrossEntropy
from utils import getMatrixOfClass,save_model,data_loader,accuracy   
from Layers import Conv2d,NonLinear,BN,Linear,Layer,Maxpool,Averagepool

class Network(Model):
    def __init__(self):
        super(Network,self).__init__()
        self.linear1 = Linear(28*28,14*14);
        self.linear2 = Linear(14*14,7*7);
        self.linear3 = Linear(7*7,10);
        self.relu = NonLinear(subtype='relu')
        
    def forward(self,input_data):
        op = self.linear3(self.relu(self.linear2(self.relu(self.linear1(input_data)))))
        return op
 

if __name__ == '__main__':

    iteration = 120000
    batch_size = 128
    save_iter = 4000
    validate_iter = 10
    disp_iter = 5
    
    traing_samples = 'train-images.idx3-ubyte' 
    traing_labels = 'train-labels.idx1-ubyte'
    test_samples = 't10k-images.idx3-ubyte'
    test_labels = 't10k-labels.idx1-ubyte'
    
    model_path = 'iteration_20.json'
    
    parameters = {
                "lr":0.001,
                "beta1":0.9,
                "beta2":0.999,
            }
     
    _iter = data_loader(traing_samples,traing_labels,batch_size)
    
    _validate = data_loader(test_samples,test_labels,1)
    
    network = Network()
    #network = Model().load(model_path)
    network.init()
   
    optimizer = Optimizer(parameters)

    criterion = CrossEntropy()
    
    for i in range(iteration):
        
        input_image, label = _iter.__next__() # 2.7 )_iter.next() / 3.6 : _iter.__next__()

        label = getMatrixOfClass(label)

        xlabel = network(np.reshape(input_image,(-1,28*28)))
        
        loss = criterion(xlabel,label)

        d_loss_network = criterion.backward(xlabel,label)
        
        network.backward(input_image,d_loss_network)
        
        optimizer(parameters)
        
        #if i % disp_iter == 0:
        #    print("iter:{},loss:{}".format(i,loss))
    
        if i % save_iter == 0:
            save_model("iteration_{}.json".format(i),network)
            
        if i % validate_iter == 0:
            # 10000
            count = 0
            for j in range(10000):
                valid_image, vlabel = _validate.__next__()
                vlabel = getMatrixOfClass(vlabel)
                vxlabel = network(np.reshape(valid_image,(-1,28*28)))   
                count = count + accuracy(vxlabel,vlabel)
            print("iteration:{}, accuracy:{}".format(i,count/10000))
