import numpy as np
from rem.Layers.Layer import Layer
from rem.Utils.utils import im2col,col2im

class Conv2d(Layer):
    def __init__(self,input_channel,output_channel,kernel,stride,padding,**kwags):
        super(Conv2d,self).__init__(**kwags)
        self.type = 'conv2d'
        self.input_channel = input_channel
        self.output_channel  = output_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = np.zeros((output_channel,input_channel,kernel[0],kernel[1]))
        self.weight_grad = np.zeros((output_channel,input_channel,kernel[0],kernel[1]))
        self.bias  = np.zeros(output_channel)
        self.bias_grad = np.zeros(output_channel)
        
    def forward(self, input_data):
        # bs x ch x w x h
        #t = time.time()
        bs,ch,h,w = input_data.shape
        
        self.input = input_data
        self.output_x = (w + 2*self.padding[0] - self.kernel[0] )//self.stride[0] + 1;
        self.output_y = (h + 2*self.padding[1] - self.kernel[1] )//self.stride[1] + 1;
        
        self.output = np.zeros( (bs,self.output_channel,self.output_y,self.output_x) ) 

        # (bs,ic*k*k,oy*ox)    
        self.imcol_all = im2col(input_data,self.kernel,self.stride,self.padding)
        
        #(oc,1)
        bs_bais = np.reshape(self.bias,[self.output_channel,1])
        
        # (oc , ic*k*k) 
        bs_weight = np.reshape(self.weight,[self.output_channel,-1]);
        
        for i in range(bs):
            #(oc,oy*ox) + (oc,1)
            M = np.dot(bs_weight,self.imcol_all[i]) + bs_bais;
            self.output[i,:,:,:] = np.reshape(M,(self.output_channel,self.output_y,self.output_x))

        return self.output
    
    def backward(self, input_data, grad_from_back):

        bs,ch,h,w = input_data.shape

        self.weight_grad.fill(0.0)
        self.grad_input = np.zeros(input_data.shape)
      
        if grad_from_back.ndim < 4: # better to throw a warning here.
            grad_from_back = np.reshape(grad_from_back, (bs,self.output_channel,self.output_y,self.output_x) )
        bs,oc,oh,ow = grad_from_back.shape
        # (bs,oc,iy,ix)
        self.bias_grad = np.sum(grad_from_back,axis=(0,2,3))
        
        #(bs, x*y, ic*k*k)
        trans_imcol = np.transpose(self.imcol_all,(0,2,1))
        # (bs, oc , oy*ox)
        for i in range(bs):
            self.weight_grad += np.reshape(np.dot(np.reshape(grad_from_back[i],(oc,oh*ow)),trans_imcol[i]),(oc,ch,self.kernel[0],self.kernel[1]))
        
        self.weight_grad = self.weight_grad/bs;
        
        d_col = np.zeros((bs,self.input_channel*self.kernel[0]*self.kernel[1],oh*ow))
        
        for i in range(bs):
            d_col[i] = np.dot(np.reshape(self.weight,(oc,-1)).T,np.reshape(grad_from_back[i],(oc,-1)))

        # reshape the d_col to image. d_col[bs,ic*k*k,oh*ow]
        self.grad_input = col2im(d_col,input_data.shape,self.kernel,self.stride,self.padding)

        return self.grad_input