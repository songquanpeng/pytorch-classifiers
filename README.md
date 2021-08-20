# PyTorch Classifiers
> PyTorch implementation of the classic classifiers network.

## Implemented Classifiers
+ [x] LeNet
+ [x] AlexNet
+ [x] VGG16
+ [ ] GoogLeNet
+ [ ] ResNet-18, ResNet-50
+ [ ] Inception-v3
+ [ ] DenseNet
+ [ ] SENet
+ [ ] EfficientNet

## Results
### Accuracy on MNIST
|Model|Parameters|Batch Size|Iterations|Accuracy|
|---|---|---|---|---|
|LeNet-5|0.0371M|64|10k|97.8404%|
|AlexNet|58.2991M|64|10k|99.0048%|
|VGG16|134.3004M|16|2k|97.6600%|

### Accuracy on CIFAR-10
|Model|Parameters|Batch Size|Iterations|Accuracy|
|---|---|---|---|---|
|LeNet-5|0.0371M|64|10k|52.1994%|
|AlexNet|58.3223M|64|10k|80.5434%|
|VGG16|134.3015M|16|2k|54.5800%|

### Accuracy on CIFAR-100
|Model|Parameters|Batch Size|Iterations|Accuracy|
|---|---|---|---|---|
|LeNet-5|0.0448M|64|10k|15.8340%|
|AlexNet|58.6910M|64|10k|48.8256%|
|VGG16|134.6702M|16|10k|43.1100%|

## Architectures
### LeNet
![arch](./assets/lenet.png)
[Image Credit](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

### AlexNet
![arch](./assets/alexnet.png)
[Image Credit](https://www.researchgate.net/figure/AlexNet-architecture-Includes-5-convolutional-layers-and-3-fullyconnected-layers_fig3_322592079)

### VGG16
![arch](./assets/vgg16.png)
[Image Credit](https://neurohive.io/en/popular-networks/vgg16/)

## Others
This repository is for practice purposes only.