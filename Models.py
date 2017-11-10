import json

class Network(object):
    def __init__(self):
        self.linear1 = Linear(28*28,14*14);
        self.linear2 = Linear(14*14,7*7);
        self.linear3 = Linear(7*7,10);
        self.relu = NonLinear(subtype='relu')
        self.stack = []

    def __repr__(self):
        re = dict()
        for k,v in self.__dict__.iteritems():
            if hasattr(self,k):
                re[k] = repr(v)
        return json.dumps(re)

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
        


