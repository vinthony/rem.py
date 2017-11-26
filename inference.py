import argparse
from rem.Models.Model import Model
from rem.Utils.utils import getMatrixOfClass,save_model,data_loader,accuracy      
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='add_model')
parser.add_argument('--folder', help='read from folder')
args = parser.parse_args()

files = [f for f in listdir(args.folder) if isfile(join(args.folder, f))]

input_shape = (-1,1,28,28)

network = Model().load(args.model)

network.evaluate()

count = 0
for j in range(len(files)):
    valid_image = Image.open(args.folder+'/'+files[j])
    valid_image = np.array(valid_image).astype('float')/255.0 
    vxlabel = network(np.reshape(valid_image,input_shape))   
    p = np.argmax(vxlabel)
    label = int(files[j].replace('.png','').split('_')[1])
    if label == p:
        re = bcolors.OKGREEN + 'True' + bcolors.ENDC
        count = count + 1;
    else:
        re = bcolors.FAIL + 'False' + bcolors.ENDC
    print(files[j],p,re)

# print('accuracy:',count/len(files))