# rem.py

a toy neural network based on python

# Todo

* layers:
    - [x] Conv2d
    - [x] Batch normalization  
    - [x] Maxpooling / Averagepooling
    - [x] relu
    - [x] linear
    - [x] dropout
    
* loss:
    - [x] L1/L2
    - [x] cross entropy
* init:
    - [x] Xavier
    - [x] Kaiming
    - [x] normal
     
* optimizer:
    - [x] adam
    - [ ] rmsprop
    - [x] sgd
    
* backend:
    - [ ] c/cpp
    - [ ] cuda
    
* tester with pytorch:
    - [ ] layers
    - [ ] optimizer
    - [ ] other

* Algorithm Examples:
    - [x] MLP
    - [x] CNN

# Before train

> download the MNIST from http://yann.lecun.com/exdb/mnist/ and create a folder name dataset to unzip the files
> you can also change the dir of file directly in `train.py`

# Easy to use

a MNIST example can be run as :

```
python train.py
```

# Easy to define the network

* you need to implement the `Network` class for your model , this class should inherit from `Model`

* you need to define the structure in the `__init__` function for use

* just run the network in the `forward` method


```
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

```

# Currently

1. Currently, I am checking the network structure from `gc`, it is ugly and hackable for feed-forward network.

I will find someway else for more complex network structure.

2. Currently, the API of design will changed heavily and the code will be optimized for faster inference. 

# Some results

default network structure(as shown above) validates on MNIST test set.

|network|   |  iteration | accuracy|
|---|----|----| --- |
| MLP(adam)[-1,1] | 1.37h |10000(~20epochs) |  97.32% |
| MLP(adam)[0,1] | 1.37h |10000(~20epochs) | 97.67% |
| MLP(adam)(bn)[0,1] | -- |10000(~20epochs) | 98.57% |
| MLP(adam)(dropout)[0,1] | 1.37h |10000(~20epochs) | 98.56% |
| CNN(adam)[0,1] | 5h |4000(~8epochs) | 98.62% |
| CNN(adam)(dropout)[0,1] | 5h |6000(~12epochs) | 99.17% |

