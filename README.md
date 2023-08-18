# PS
This repository contains the author's implementation in Pytorch for method PS of SFDA method.

## Environment Requirement
- python 3.6.5
- pytorch 1.6.0
- torchvision 0.7.0
- cuda 10.1
- numpy, sklearn, scipy

# Data
We use VisDA dataset in this code. Please download the dataset at [VisDA dataset](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) from the official websites, and modify the path of images in each '.txt' under the folder './data/'.

# Training
Firstly, you need to train the source model by running this code
```
python train_source.py
```
Then, you can run the code of PS to perform SFDA,
```
python adaptation.py
```
