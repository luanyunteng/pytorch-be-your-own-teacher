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
python train.py train multi_resnet18_kd \
                             --data-dir /PATH/TO/CIFAR100 
```

## Load an ResNet on CIFAR-100 and test it:
```
python train.py test multi_resnet18_kd \
                             --data-dir /PATH/TO/CIFAR100 
                             --resume /PATH/TO/checkpoint
```

## Result on Resnet18

As the original paper does not tell us the hyper-parameters, I just use the follow setting. If you find better hyper-parmeters, you could tell me in the issues. Moreover, we do not konow the side-branch architecture details. So the accuracy is lower than the original paper.
I will fine-tune the hyper-parameters.
# alpha = 0.1
# temperature = 3
# beta = 1e-6

Method | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 
-------  | ------- | ------- ｜ ------- ｜ -------
Original | 67.85   | 74.57   ｜ 78.23   ｜ 78.64
Ours     | 67.17   | 73.27    | 77.14   | 77.86


