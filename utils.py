import numpy as np
import struct

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
    r = k[0]//2;
    c = k[1]//2;
    
    sr = stride[0];
    sc = stride[1];
    
    pr = padding[0];
    pc = padding[1];

    h,w = imgray.shape
    
    re = np.zeros(r*c,(h-1)*stride * (w-1)*stride);
    for y in range(h):
        for x in range(w):
            # if border, padding
            if y == 0 or x == 0 or y == h-1 or x == w-1:
                # re[] = 
                print('border')
            else:
                re[:,w*(x-1)+h] = imgray[y-r:y+r,x-r:x+c]
    
    return re;
