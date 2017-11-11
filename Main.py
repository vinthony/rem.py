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
from Layers import Conv2d,NonLinear,BN,Linear,Layer,Maxpool,Averagepool,Dropout

class NetworkWithDropout(Model):
    def __init__(self):
        super(NetworkWithDropout,self).__init__()
        self.linear1 = Linear(28*28,14*14);
        self.linear2 = Linear(14*14,7*7);
        self.linear3 = Linear(7*7,10);
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.5)
        self.relu1 = NonLinear(subtype='relu')
        self.relu2 = NonLinear(subtype='relu')
        
    def forward(self,input_data):
        op = self.linear3(self.dropout2(self.relu2(self.linear2(self.dropout1(self.relu1(self.linear1(input_data)))))))
        return op
    
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
    
class NetworkWithBN(Model):
    def __init__(self):
        super(NetworkWithBN,self).__init__()
        self.linear1 = Linear(28*28,14*14);
        self.linear2 = Linear(14*14,7*7);
        self.linear3 = Linear(7*7,10);
        self.relu1 = NonLinear(subtype='relu')
        self.relu2 = NonLinear(subtype='relu')
        self.BN1 = BN(14*14)
        self.BN2 = BN(7*7)
        
    def forward(self,input_data):
        op = self.linear3(
                self.relu1(self.BN2(self.linear2(
                self.relu2(self.BN1(self.linear1(input_data)))
                )))
                )
        return op

class CNN(Model):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = Conv2d(1,32,[3,3],[2,2],[1,1]); # 28x28 -> 32 x 14 x 14
        self.conv2 = Conv2d(32,32,[3,3],[2,2],[1,1]); # 32 x 7 x 7
        self.conv3 = Conv2d(32,32,[3,3],[2,2],[1,1]); # 32 x 4 x 4
        # self.bn1 = BN(32)
        # self.bn2 = BN(32)
        # self.bn3 = BN(64)
        # self.bn4 = BN(64)
        # self.bn5 = BN(128)
        self.linear1 = Linear(512,49);
        self.linear2 = Linear(49,10); # 
        self.relu1 = NonLinear(subtype='relu')
        self.relu2 = NonLinear(subtype='relu')
        self.relu3 = NonLinear(subtype='relu')
        self.relu4 = NonLinear(subtype='relu')
        
    def forward(self,input_data):
        op1 = self.linear2(self.relu4(self.linear1(self.relu3(self.conv3(self.relu2(self.conv2(self.relu1(self.conv1(input_data)))))))))
        return op1
 
 

if __name__ == '__main__':

    iteration = 120000
    batch_size = 256
    save_iter = 1000
    validate_iter = 20
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
    init_type = 'xavier' # console4:kaiming
     
    _iter = data_loader(traing_samples,traing_labels,batch_size)
    
    _validate = data_loader(test_samples,test_labels,1)
    
    network = NetworkWithDropout()
    #network = Model().load(model_path)
    network.init(init_type)
    
    print('init complete')
   
    optimizer = Optimizer(parameters)

    criterion = CrossEntropy()
    
    for i in range(iteration):
        
        input_image, label = _iter.__next__() # 2.7 )_iter.next() / 3.6 : _iter.__next__()

        label = getMatrixOfClass(label)

        xlabel = network(np.reshape(input_image,(-1,1,28,28)))
        
        loss = criterion(xlabel,label)

        d_loss_network = criterion.backward(xlabel,label)
        
        network.backward(input_image,d_loss_network)
        
        optimizer(parameters)
        
    
        if i % save_iter == 0:
            save_model("iteration_{}_{}.json".format(i,init_type),network)
            
        if (i+1) % validate_iter == 0:
            # 10000
            count = 0
            for j in range(10000):
                valid_image, vlabel = _validate.__next__()
                vlabel = getMatrixOfClass(vlabel)
                vxlabel = network(np.reshape(valid_image,(-1,1,28,28)))   
                count = count + accuracy(vxlabel,vlabel)
            print("iter:{},loss:{}, accuracy:{}".format(i,loss,count/10000))
