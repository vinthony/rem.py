import argparse
import numpy as np
from rem.Models.Model import Model
from rem.Utils.utils import getMatrixOfClass,save_model,data_loader,accuracy   

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='add_model')
args = parser.parse_args()

test_samples = 'dataset/t10k-images-idx3-ubyte'
test_labels = 'dataset/t10k-labels-idx1-ubyte'

_validate = data_loader(test_samples,test_labels,1,False)

input_shape = (-1,1,28,28)

network = Model().load(args.model)

network.evaluate()

count = 0
for j in range(10000):
    valid_image, vlabel = _validate.__next__()
    vlabel = getMatrixOfClass(vlabel)
    vxlabel = network(np.reshape(valid_image,input_shape))   
    count = count + accuracy(vxlabel,vlabel)
print("accuracy:{:.2f}%".format(count/10000*100))
