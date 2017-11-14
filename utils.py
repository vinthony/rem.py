import numpy as np
import struct
import math

def save_model(filename,network):
    with open(filename, 'w') as f:
        network.save(f)

def getMatrixOfClass(target_data,labels=10):
    bs, = target_data.shape
    mx = np.zeros((bs,10))
    idx = np.arange(bs)
    mx[idx[:],target_data[:]] = 1;
    return mx

def data_loader(image,label,batch_size):
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
            batch_image = images[batch_idx:batch_idx+batch_size].astype('float32')
            #normalize
            batch_image = (batch_image/255.0)
        
            yield batch_image,batch_label
            
def accuracy(input_label, target_label):
    input_label = np.reshape(input_label,(1,-1))
    target_label = np.reshape(target_label,(1,-1))
    il = np.argmax(input_label);
    tl = np.argmax(target_label)
    return il == tl

def im2col(imgray,k,stride,padding):
    #  im2col 
    bs,ch,h,w = imgray.shape
    r = k[0];
    c = k[1];
    
    sr = stride[0];
    sc = stride[1];
    
    pr = padding[0];
    pc = padding[1];
    
    output_x = (w + 2*pr - r )//sr + 1;
    output_y = (h + 2*pc - c )//sc + 1;

    larger = np.zeros((bs,ch,h+2*pr,w+2*pc))

    # padding
    larger[:,:,pr:pr+h,pc:pc+w] = imgray
    
    re = np.zeros( (bs,ch*r*c,output_x*output_y),dtype='float32');
    count = 0
    for x in range(0,w+2*pc-c+1,sc):
        for y in range(0,h+2*pr-r+1,sr):
            assert y+r-1 < h+2*pr
            assert x+c-1 < w+2*pc
            re[:,:,count] = np.reshape(larger[:,:,y:y+r,x:x+c],(bs,-1))
            count = count + 1
    assert count == output_x*output_y
    return re;

def col2im(d_col,shape_x,k,stride,padding):
    bs,ch,h,w = shape_x
    bs,ickk,ohow = d_col.shape
    r = k[0];
    c = k[1];
    
    sr = stride[0];
    sc = stride[1];
    
    pr = padding[0];
    pc = padding[1];

    re = np.zeros((bs,ch,h+2*pr,w+2*pc),dtype='float32')

    count = 0
    for x in range(0,w+2*pc-c+1,sc):
        for y in range(0,h+2*pr-r+1,sr):
            assert y+r-1 < h+2*pr
            assert x+c-1 < w+2*pc
            re[:,:,y:y+r,x:x+c] += np.reshape(d_col[:,:,count],(bs,ch,k[0],k[1]))
            count = count + 1
    assert count == ohow
    return re[:,:,pr:-pr,pc:-pc]



def col2im_x(imgray,k,stride,padding,adjr,adjc):
    bs,ch,h,w = imgray.shape
    r = k[0];
    c = k[1];
    
    sr = stride[0];
    sc = stride[1];
    
    pr = padding[0];
    pc = padding[1];
    output_x = (w-1)*sr - 2*pr + r + adjr
    output_y = (h-1)*sc - 2*pc + c + adjc
    larger = np.zeros((bs,ch,output_y+2*pr,output_x+2*pc))

    ix = np.arange(pc,pc+output_x,sc)
    iy = np.arange(pr,pr+output_y,sr)

    xv, yv = np.meshgrid(ix, iy)
    
    larger[:,:,yv,xv] = imgray

    re = np.zeros( (bs,ch*r*c,output_x*output_y),dtype='float32');

    count = 0
    for x in range(0,output_x+2*pc-c+1):
        for y in range(0,output_y+2*pr-r+1):
            re[:,:,count] = np.reshape(larger[:,:,y:y+r,x:x+c],(bs,-1))
            count = count + 1
    assert count == output_x*output_y
    return re


if __name__ == '__main__':
    im = np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[5,6,7,8],[1,2,3,4],[9,10,11,12]],[[5,6,7,8],[1,2,3,4],[9,10,11,12]]],[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[1,2,3,4],[5,6,7,8],[9,10,11,12]]]]);
    kernel = [2,2]
    stride = [1,1]
    padding = [1,1]
    print(im.shape)
    out = im2col(im,kernel,stride,padding)
    print(out)
    out2=col2im(out,im.shape,kernel,stride,padding)
    print(out2)
