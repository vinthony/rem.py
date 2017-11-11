import numpy as np
import struct
import math

def save_model(filename,network):
    # save the weight of network;
    # save the structure of network;
    #f = h5py.File(filename,"w")
    #f.create_dataset('network',  len(repr(network)) ,'s10',repr(network))
    with open(filename, 'w') as f:
        network.save(f)

def getMatrixOfClass(target_data,labels=10):
    bs, = target_data.shape
    mx = np.zeros((bs,10))
    idx = np.arange(bs)
    mx[idx[:],target_data[:]] = 1;
    return mx

def data_loader(image,label,batch_size):
    # generater an image from datasert
    with open(label,'rb') as lb:
        # file header [0,9]
        m,n = struct.unpack('>II',lb.read(8))
        labels = np.fromfile(lb, dtype=np.uint8)

    with open(image,'rb') as im:
        m,n,r,c = struct.unpack('>IIII',im.read(16))
        images = np.fromfile(im,dtype=np.uint8).reshape(len(labels),28,28)

    length = len(labels)

    while True:
        idxs = np.arange(length)
        
        if batch_size > 1:
            np.random.shuffle(idxs)

        for batch_idx in range(0,length,batch_size):

            batch_label = labels[batch_idx:batch_idx+batch_size]
            batch_image = images[batch_idx:batch_idx+batch_size]
            
            yield batch_image,batch_label
            
def accuracy(input_label, target_label):
    input_label = np.reshape(input_label,(1,-1))
    target_label = np.reshape(target_label,(1,-1))
    il = np.argmax(input_label);
    tl = np.argmax(target_label)
    return il == tl

def im2col(imgray,k,stride,padding):
    h,w = imgray.shape

    r = k[0];
    c = k[1];
    
    sr = stride[0];
    sc = stride[1];
    
    pr = padding[0];
    pc = padding[1];

    output_x = (w + 2*pr - r )//sr + 1;
    output_y = (h + 2*pc - c )//sc + 1;
    
    larger = np.zeros((h+2*pr,w+2*pc))

    larger[pr:pr+h,pc:pc+w] = imgray
    
    re = np.zeros( (r*c,output_x*output_y),dtype='float32');

    for y in range(0,h+2*pr-r+1,sr):
        for x in range(0,w+2*pc-c+1,sc):
            re[:,output_x*(y//sr)+(x//sc)] = np.reshape(larger[y:y+r,x:x+c],(1,-1))
    
    return re;


def imremap(imgray,oh,ow,k,stride,padding):
    h,w = imgray.shape

    r = k[0];
    c = k[1];
    
    sr = stride[0];
    sc = stride[1];
    
    pr = padding[0];
    pc = padding[1];

    output_x = math.ceil( (w-1)*sr + r - 2*pr ); #(w + 2*pr - r )//sr + 1;
    output_y = math.ceil( (h-1)*sc + c - 2*pc ); #(h + 2*pc - c )//sc + 1;
    
    if output_x < ow: output_x = ow;
    if output_y < oh: output_y = oh;

    larger = np.zeros((h+2*pr,w+2*pc))

    larger[pr:pr+h,pc:pc+w] = imgray
    
    re = np.zeros( (r*c,output_x*output_y),dtype='float32');

    for y in range(0,h+2*pr-r+1,sr):
        for x in range(0,w+2*pc-c+1,sc):
            re[:,output_x*(y*sr)+(x*sc)] = np.reshape(larger[y:y+r,x:x+c],(1,-1))

    return re;


if __name__ == '__main__':
    im = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]);
    kernel = [2,2]
    stride = [1,1]
    padding = [1,1]
    out = im2col(im,kernel,stride,padding)
