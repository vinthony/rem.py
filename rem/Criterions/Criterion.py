class Criterion(object):
    def __init__(self):
        pass
    
    def __call__(self,input_data,target_data):
        return self.forward(input_data,target_data)
    
    def forward(self,input_data,target_data):
        pass
    
    def backward(self,input_data,target_data):
        pass