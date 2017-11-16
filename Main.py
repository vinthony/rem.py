# -*- coding: utf-8 -*-
"""
Spyder Editor 2017年11月2日23:41:00

This is a temporary script file.
"""

import numpy as np
import time
import datetime
from Models import Model
import logging

from Optimizer import Optimizer
from Criterions import CrossEntropy
from utils import getMatrixOfClass,save_model,data_loader,accuracy   
from Layers import Conv2d,NonLinear,BN,Linear,Layer,Maxpool,Averagepool,Dropout,SpatialBN

class MLPWithDropout(Model):
    def __init__(self):
        super(MLPWithDropout,self).__init__()
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

class MLPKeras(Model):
    def __init__(self):
        super(MLPKeras,self).__init__()
        self.linear1 = Linear(28*28,512);
        self.linear2 = Linear(512,512);
        self.linear3 = Linear(512,10);
        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.relu1 = NonLinear(subtype='relu')
        self.relu2 = NonLinear(subtype='relu')
        
    def forward(self,input_data):
        op = self.linear3(self.dropout2(self.relu2(self.linear2(self.dropout1(self.relu1(self.linear1(input_data)))))))
        return op

class MLP(Model):
    def __init__(self):
        super(MLP,self).__init__()
        self.linear1 = Linear(28*28,14*14);
        self.linear2 = Linear(14*14,7*7);
        self.linear3 = Linear(7*7,10);
        self.relu1 = NonLinear(subtype='relu')
        self.relu2 = NonLinear(subtype='relu')
        
    def forward(self,input_data):
        op = self.linear3(self.relu1(self.linear2(self.relu2(self.linear1(input_data)))))
        return op
    
class MLPWithBN(Model):
    def __init__(self):
        super(MLPWithBN,self).__init__()
        self.linear1 = Linear(28*28,14*14);
        self.linear2 = Linear(14*14,7*7);
        self.linear3 = Linear(7*7,10);
        self.relu1 = NonLinear(subtype='relu')
        self.relu2 = NonLinear(subtype='relu')
        self.BN1 = BN()
        self.BN2 = BN()
        
    def forward(self,input_data):
        op = self.linear3(
                self.relu1(self.BN2(self.linear2(
                self.relu2(self.BN1(self.linear1(input_data)))
                )))
                )
        return op


class CNNWithBN(Model):
    def __init__(self):
        super(CNNWithBN,self).__init__()
        self.conv1 = Conv2d(1,32,[3,3],[2,2],[1,1]); # 28x28 -> 32 x 14 x 14
        self.conv2 = Conv2d(32,32,[3,3],[2,2],[1,1]); # 32x14x14 -> 32 x 7 x 7
        self.conv3 = Conv2d(32,32,[3,3],[2,2],[1,1]); # 32 x 4 x 4
        self.bn1 = SpatialBN(32)
        self.bn2 = SpatialBN(32)
        self.bn3 = SpatialBN(32)
        # self.bn5 = BN(128)
        self.linear1 = Linear(512,49);
        self.linear2 = Linear(49,10); # 
        self.relu1 = NonLinear(subtype='relu')
        self.relu2 = NonLinear(subtype='relu')
        self.relu3 = NonLinear(subtype='relu')
        self.relu4 = NonLinear(subtype='relu')
        
    def forward(self,input_data):
        op1 = self.relu1(self.bn1(self.conv1(input_data)));
        op2 = self.relu2(self.bn2(self.conv2(op1)));
        op3 = self.linear1(self.bn3(self.relu3(self.conv3(op2))));
        op4 = self.linear2(self.relu4(op3))
        return op4


class CNNKerasDropout(Model):
    def __init__(self):
        super(CNNKerasDropout,self).__init__()
        self.conv1 = Conv2d(1,32,[3,3],[1,1],[1,1]); # 28x28 -> 32 x 28 x 28
        self.conv2 = Conv2d(32,64,[3,3],[2,2],[1,1]); # 32x14x14 -> 64 x 7 x 7
#        self.bn1 = SpatialBN(32)
#        self.bn2 = SpatialBN(32)
#        self.bn3 = SpatialBN(32)
        # self.bn5 = BN(128)
        self.dropout1 = Dropout(0.25)
        self.dropout2 = Dropout(0.5)
        self.linear1 = Linear(12544,128);
        self.linear2 = Linear(128,10); # 
        self.relu1 = NonLinear(subtype='relu')
        self.relu2 = NonLinear(subtype='relu')
        self.relu4 = NonLinear(subtype='relu')
        
    def forward(self,input_data):
        op1 = self.relu1(self.conv1(input_data));
        op2 = self.dropout1(self.relu2(self.conv2(op1)));
        op3 = self.dropout2(self.relu4(self.linear1(op2)));
        op4 = self.linear2(op3)
        return op4  

class CNNKeras(Model):
    def __init__(self):
        super(CNNKeras,self).__init__()
        self.conv1 = Conv2d(1,32,[3,3],[1,1],[1,1]); # 28x28 -> 32 x 28 x 28
        self.conv2 = Conv2d(32,64,[3,3],[2,2],[1,1]); # 32x14x14 -> 64 x 7 x 7

        self.linear1 = Linear(12544,128);
        self.linear2 = Linear(128,10); # 
        self.relu1 = NonLinear(subtype='relu')
        self.relu2 = NonLinear(subtype='relu')
        self.relu4 = NonLinear(subtype='relu')
        
    def forward(self,input_data):
        op1 = self.relu1(self.conv1(input_data));
        op2 = self.relu2(self.conv2(op1));
        op3 = self.linear1(op2);
        op4 = self.linear2(self.relu4(op3))
        return op4 

    
class CNN(Model):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = Conv2d(1,32,[3,3],[2,2],[1,1]); # 28x28 -> 32 x 14 x 14
        self.conv2 = Conv2d(32,32,[3,3],[2,2],[1,1]); # 32x14x14 -> 32 x 7 x 7
        self.conv3 = Conv2d(32,32,[3,3],[2,2],[1,1]); # 32 x 4 x 4
#        self.bn1 = SpatialBN(32)
#        self.bn2 = SpatialBN(32)
#        self.bn3 = SpatialBN(32)
        # self.bn5 = BN(128)
        self.linear1 = Linear(512,49);
        self.linear2 = Linear(49,10); # 
        self.relu1 = NonLinear(subtype='relu')
        self.relu2 = NonLinear(subtype='relu')
        self.relu3 = NonLinear(subtype='relu')
        self.relu4 = NonLinear(subtype='relu')
        
    def forward(self,input_data):
        op1 = self.relu1(self.conv1(input_data));
        op2 = self.relu2(self.conv2(op1));
        op3 = self.linear1(self.relu3(self.conv3(op2)));
        op4 = self.linear2(self.relu4(op3))
        return op4
 
 

if __name__ == '__main__':

    iteration = 10000 # ~20epochs
    batch_size = 128
    save_iter = 50
    validate_iter = 50
    
    traing_samples = 'dataset/train-images.idx3-ubyte' 
    traing_labels = 'dataset/train-labels.idx1-ubyte'
    test_samples = 'dataset/t10k-images.idx3-ubyte'
    test_labels = 'dataset/t10k-labels.idx1-ubyte'
    
    model_path = 'iteration_20.json'

    name = time.time()

    logging.basicConfig(filename='checkpoints/{}.log'.format(name),level=logging.INFO)
    
    input_shape = (-1,1,28,28)

    #adam
    parameters = {
                "lr":0.001,
                "beta1":0.9,
                "beta2":0.999,
            }
    init_type = 'xavier' # normal,kaiming,xavier
     
    _iter = data_loader(traing_samples,traing_labels,batch_size)
    
    _validate = data_loader(test_samples,test_labels,1)
    
    network = CNNKerasDropout()

    network.init(init_type)
    
    optimizer = Optimizer(parameters)

    criterion = CrossEntropy()
    
    begin = time.time()
    best_acc = 0  
    logs = '[{}][network:{}][optimizer:{}][learningRate:{}][init_method:{}]'.format(datetime.datetime.fromtimestamp(name),network.__class__.__name__,'Adam',parameters['lr'],init_type)
    logging.info(logs)
    print(logs)
    for i in range(iteration):
        
        input_image, label = _iter.__next__() # 2.7 )_iter.next() / 3.6 : _iter.__next__()

        label = getMatrixOfClass(label)

        xlabel = network(np.reshape(input_image,input_shape))

        loss = criterion(xlabel,label)

        d_loss_network = criterion.backward(xlabel,label)
        
        network.backward(input_image,d_loss_network)
        
        optimizer(parameters)
        
        if i % save_iter == 0:
            save_model("checkpoints/iteration_n_{}_{}.json".format(i,init_type),network)
            
        if (i+1) % validate_iter == 0:
            network.evaluate()
            # 10000
            count = 0
            for j in range(10000):
                valid_image, vlabel = _validate.__next__()
                vlabel = getMatrixOfClass(vlabel)
                vxlabel = network(np.reshape(valid_image,input_shape))   
                count = count + accuracy(vxlabel,vlabel)
            if count/10000*100 > best_acc:
                best_acc = count/10000*100
                save_model("checkpoints/best_{}_{}.json".format(i,init_type),network)
            logging.info("iter:{},loss:{:.2f}, accuracy:{:.2f}%,time:{:.2f}h".format(i,loss,count/10000*100,(time.time()-begin)/3600))
            network.train()

    logging.info("total time:{:.2f}h, best accuracy:{:.2f}%, model paramaters:{}".format((time.time()-begin)/3600,best_acc,network.getParametersOfModel()))
