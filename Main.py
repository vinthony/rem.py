# -*- coding: utf-8 -*-
"""
Spyder Editor 2017年11月2日23:41:00

This is a temporary script file.
"""

import json
import numpy as np
import struct
import gc
import h5py


from Models import Network
from Layers import Conv2d,NonLinear,BN,Linear,Layer,Maxpool,Averagepool
from Optimizer import Optimizer
from Criterions import CrossEntropy
from utils import getMatrixOfClass,save_model,data_loader   


if __name__ == '__main__':

    iteration = 120000
    batch_size = 128
    save_iter = 1000
    
    traing_samples = 'train-images.idx3-ubyte' 
    traing_labels = 'train-labels.idx1-ubyte'

    parameters = {
                "lr":0.001,
                "beta1":0.9,
                "beta2":0.999,
            }
     
    _iter = data_loader(traing_samples,traing_labels,batch_size)
    
    network = Network()

    network.init()
   
    optimizer = Optimizer(parameters)

    criterion = CrossEntropy()
    
    for i in range(iteration):
        
        input_image, label = _iter.next()

        label = getMatrixOfClass(label)

        xlabel = network(np.reshape(input_image,(-1,28*28)))
        
        loss = criterion(xlabel,label)

        print('iter:',i,loss)

        d_loss_network = criterion.backward(xlabel,label)
        
        network.backward(input_image,d_loss_network)
        
        optimizer(parameters)

    if i % save_iter == 0:
        f = h5py.File("iteration_{}.h5".format(i),"w")
        save_model(f,network)
