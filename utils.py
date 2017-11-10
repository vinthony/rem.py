import numpy as np

def save_model(f,network):
    # save the weight of network;
    # save the structure of network;
    f.create_dataset('network',(len(repr(network))),'s10',repr(network))
    stack = network.get_stack()
    re = dict()
    for i in stack:
        re[i.type+''+i.callid+'_w'] = i.get_weights()
        re[i.type+''+i.callid+'_b'] = i.get_bias()
    f.create_dataset('weight',len(json.dump(re)),'s10',json.dump(re))

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
        np.random.shuffle(idxs)

        for batch_idx in range(0,length,batch_size):

            batch_label = labels[batch_idx:batch_idx+batch_size]
            batch_image = images[batch_idx:batch_idx+batch_size]
            
            yield batch_image,batch_label