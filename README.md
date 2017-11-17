# rem.py

a toy neural network based on python

# todo

* layers:
    - [x] Conv2d
    - [x] Batch normalization (some bugs here)
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

# easy to use

a MNIST example can be run as :

```
python Main.py
```

# easy to define the network

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

# currently

currently, I am checking the network structure from `gc`, it is ugly and hackable for feed-forward network.

I will find someway else for more complex network structure.


# some results

default network structure(as shown above) validates on MNIST test set.

|network| iteration | time | accuracy|
|---|----|----| --- |
| MLP(sgd) | -- | 10000(~20epochs) | 85%|
| MLP(adam)[-1,1] | 4768s |10000(~20epochs) |  97.32% |
| MLP(adam)[0,1] | 4768s |10000(~20epochs) | 97.67% |
| MLP(adam)(bn)[0,1] | 4768s |10000(~20epochs) | 98.57% |
| MLP(adam)(dropout)[0,1] | 4768s |10000(~20epochs) | 98.57% |
| CNN(adam)[0,1] | -- |4000(~8epochs) | 98.62% |
| CNN(adam)(dropout)[0,1] | -- |4000(~8epochs) | 99.16% |

