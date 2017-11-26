import argparse
from rem.Models.Model import Model
from rem.Utils.utils import getMatrixOfClass,save_model,data_loader,accuracy      
from os import listdir
from os.path import isfile, join
from PIL import Image

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
	valid_image = Image.open(j)
	vxlabel = network(np.reshape(valid_image,input_shape))   
	p = np.argmax(vxlabel)
	print(j,p)