import numpy as np
from rem.Layers.Layer import Layer
from rem.Utils.utils import im2col,col2im

class Maxpool(Layer):
    def __init__(self,kernel,stride,padding):
        super(Maxpool,self).__init__()
        self.type = 'maxpool'
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        
    def forward(self,input_data):
        bs,ch,h,w = input_data.shape
        
        self.input = input_data
        self.output_x = (w + 2*self.padding[0] - self.kernel[0] )//self.stride[0] + 1;
        self.output_y = (h + 2*self.padding[1] - self.kernel[1] )//self.stride[1] + 1;
         
        #bs x ic x k*k x oy*ox
        self.output = np.zeros( (bs,self.output_channel,self.output_y,self.output_x) )     
        self.imcol_all = np.reshape(im2col(input_data,self.kernel,self.stride,self.padding),(bs,ch,self.kernel[0]*self.kernel[1],self.output_y,self.output_x))
        
        # bs,op,
        self.output = np.max(self.imcol_all,2)
        self.max_idx = np.argmax(self.imcol_all,2)
        
        return self.output
    
    def backward(self,grad_from_back):
        bs,oc,h,w = grad_from_back.shape
        #(bs,self.input_channel*self.kernel[0]*self.kernel[1],oh*ow)
        #bs ic h*w
        d_col = np.zeros((bs,ic*self.kernel[0]*self.kernel[1],h*w))
        d_col[self.max_idx] = grad_from_back 
        self.grad_input=col2im(d_col,self.input.shape,self.kernel,self.stride,self.padding)
        return self.grad_input