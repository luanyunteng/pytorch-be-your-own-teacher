# pytorch-be-your-own-teacher
An pytorch implementation of paper 'Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation', https://arxiv.org/abs/1905.08094

## Introduction
We provide code of training ResNet18 and ResNet50 with multiple breanches on CIFAR100 dataset. 

## Dependencies:

+ Ubuntu 18.04.5 LTS
+ Python 3.6.9
+ PyTorch 1.7.1
+ torchvision 0.8.2 
+ numpy 1.19.2 
+ tensorboardX 2.1

Note: this is my machine environment, and the other version of software may also works.

## Train an ResNet on CIFAR-100:

```sh
# resnet18
python train.py train multi_resnet18_kd \
                             --data-dir /PATH/TO/CIFAR100 

# resnet50
python train.py train multi_resnet18_kd \
                             --data-dir /PATH/TO/CIFAR100 
```

## Load an ResNet on CIFAR-100 and test it:
```sh
# resnet18
python train.py test multi_resnet18_kd \
                             --data-dir /PATH/TO/CIFAR100 \
                             --resume /PATH/TO/checkpoint

# resnet50
python train.py test multi_resnet50_kd \
                             --data-dir /PATH/TO/CIFAR100 \
                             --resume /PATH/TO/checkpoint
```

## Result on Resnet18

As the original paper does not tell us the hyper-parameters, I just use the follow setting. If you find better hyper-parmeters, you could tell me in the issues. Moreover, we do not know the side-branch architecture details. So the accuracy is lower than the original paper.
I will fine-tune the hyper-parameters.

21.01.12) At the same settings, with different library version, the performance of last classifier is better than paper result.
However, hyperparameter searching is still needed.

 ```
alpha = 0.1
temperature = 3
beta = 1e-6
```
|   Method   | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 |
|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Original   |    67.85       |74.57          |       78.23   |     78.64      |
| Ours       | 66.570         | 74.690        | 78.040        | 78.730         |

## Result on Resnet50

 ```
alpha = 0.1
temperature = 3
beta = 1e-6
```
|   Method   | Classifier 1/4 | Classifier 2/4 | Classifier 3/4 | Classifier 4/4 |
|:----------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Original   |    68.23       |74.21          |       75.23   |     80.56      |
| Ours       |    71.840       |   77.750     |   80.140      |     80.640     |
   