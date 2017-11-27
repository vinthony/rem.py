.PHONY: init
init:
	mkdir dataset
	mkdir pretrained
	wget -O best_CNN.json.zip http://vinthony.u.qiniudn.com/best_CNN.json.zip && unzip -o best_CNN.json.zip -d pretrained/ && rm best_CNN.json.zip
	wget -O dataset/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz && gunzip dataset/train-images-idx3-ubyte.gz
	wget -O dataset/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz && gunzip dataset/train-labels-idx1-ubyte.gz
	wget -O dataset/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz && gunzip dataset/t10k-images-idx3-ubyte.gz
	wget -O dataset/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz && gunzip dataset/t10k-labels-idx1-ubyte.gz
	mkdir checkpoint
	pip install -r requirements.txt
.PHONY: clean
clean:
	rm -rf dataset
	rm -rf checkpoint
	rm -rf pretrained
