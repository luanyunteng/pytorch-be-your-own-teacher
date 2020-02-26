# pytorch-be-your-own-teacher
An pytorch implementation of paper 'Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation', https://arxiv.org/abs/1905.08094

## Introduction
We provide code of training ResNet18 and ResNet50 with multiple breanches on CIFAR100 dataset. 

## Dependencies:

+ Python 3.5.2
+ PyTorch 1.2.0
+ torchvision          0.4.0  
+ numpy 1.17.2 
+ tensorboardX         1.8

Note: this is my machine environment, and the other version of software may also works.

## Train an ResNet on CIFAR-100:

```
python train_resnet.py train multi_resnet50_kd \
                             --data-dir /PATH/TO/CIFAR100 
```

## Load an ResNet on CIFAR-100 and test it:
```
python train_resnet.py test multi_resnet50_kd \
                             --data-dir /PATH/TO/CIFAR100 
                             --resume /PATH/TO/checkpoint
```