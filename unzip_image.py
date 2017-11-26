import argparse
import numpy as np
from PIL import Image
from rem.Models.Model import Model
from rem.Utils.utils import getMatrixOfClass,save_model,data_loader,accuracy   

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='add_model')
args = parser.parse_args()

test_samples = 'dataset/t10k-images.idx3-ubyte'
test_labels = 'dataset/t10k-labels.idx1-ubyte'

_validate = data_loader(test_samples,test_labels,1,False)

count = 0
for j in range(10000):
    valid_image, vlabel = _validate.__next__()
    re= Image.fromarray((valid_image[0][0]*255).astype(np.uint8))
    re.save('test_set/{}_{}.png'.format(j,vlabel[0]))