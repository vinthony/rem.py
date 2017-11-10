

class Optimizer(object):
    
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