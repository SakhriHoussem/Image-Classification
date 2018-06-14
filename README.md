# Classification of Satellite images

## Pre-trained VGG-Net Model For Classification of Satellite images :rocket: using Tensorflow

###  DataSets :
we used each of this DataSets for training

 - [UC Merced Land Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
 - [SIRI-WHU](http://www.lmars.whu.edu.cn/prof_web/zhongyanfei/e-code.html)
 - [RSSCN7](http://www.lmars.whu.edu.cn/xia/AID-project.html)
 
## After Training :
Resultat of UC Merced Land DataSet After Training

Testing the classification of one batch of Pictures from [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) 

![test UC Merced Land](images/test19_UCMerced_LandUse.png ' classification of batch of pictre after Training')

#### `Cost` and `Accuracy` : 
graph represent the values of both of `cost` and `accuracy` each epoch
![graph](images/acc19_cost19_UCMerced_LandUse.png 'graph of Accuracy and cost')

## How To use :
you can use this model to classify any DataSet just follow the 4 next instruction

### Instalation :

* install [tensorflow 1.6](tensorflow.org) [matplotlib](https://matplotlib.org/) [opencv](https://pypi.org/project/opencv-python/) [imutils](https://pypi.org/project/imutils)
```python
 pip install tensorflow matplotlib opencv-python imutils
 ```
* to install [tensorflow gpu](https://github.com/SakhriHoussem/How-to-install-tensorflow-gpu) [matplotlib](https://matplotlib.org/) [opencv](https://pypi.org/project/opencv-python/) 
### Train the Model:
To Train Model for different DataSets or Different Classification follow the steps : 

#### Exploit your DataSet
```shell
python dataSetGenerator.py  [-h] --path path [--SaveTo SaveTo] [--resize resize]
                      [--resize_to resize_to] [--percentage percentage]
                      [--dataAug dataAugmentation]
```

##### Example
```shell
python dataSetGenerator.py  --path Desktop/SIRI-WHU --resize --resize_to 200
```

##### Help
```shell
image dataSet as numpy file.

       picture dataSets
          |
          |----------class-1
          |        .   |-------image-1
          |        .   |         .
          |        .   |         .
          |        .   |         .
          |        .   |-------image-n
          |        .
          |-------class-n

optional arguments:
  -h, --help            show this help message and exit
  --path path           the path for picture dataSets folder (/)
  --SaveTo SaveTo       the path when we save dataSet (/)
  --resize resize       choose resize the pictures or not
  --resize_to resize_to
                        the new size of pictures
  --percentage percentage
                        how many pictures you want to use for training
  --dataAug dataAugmentation
                        apply data Augmentation Strategy
 ```
#### Train your DataSet
```shell
python   train_vgg19.py [-h] --dataset dataset [--batch batch] [--epochs epochs]
```

##### Example
```shell
python train_vgg19.py  --dataset SIRI-WHU
```

##### Help
```shell
Train vgg19 [-h] --dataset dataset [--batch batch] [--epochs epochs]

Simple tester for the vgg19_trainable

optional arguments:
  -h, --help         show this help message and exit
  --dataset dataset  DataSet Name
  --batch batch      batch size
  --epochs epochs    number of epoch to train the network
```
### Test your Model :
to test your model 
```shell
python test_vgg19.py [-h] --dataset dataset [--batch batch]
```

##### Example
```shell
python test_vgg19.py  --dataset SIRI-WHU
```

#### Help
```shell
tester for the vgg19_trainable

optional arguments:
  -h, --help         show this help message and exit
  --dataset dataset  DataSet Name
  --batch batch      batch size
```
### Confusion Matrix : 
to Draw Confusion matrix (the output in images)
```shell
python confusion_matrix.py -h [-h] --dataset dataset [--batch batch] [--showPic showPic]
```
#### Help
```shell
Draw Confusion Matrix for the vgg19

optional arguments:
  -h, --help         show this help message and exit
  --dataset dataset  DataSet Name
  --batch batch      batch size
  --showPic showPic  Show patch of picture each epoch
```

### For Distributed Tensorflow [optional] : 

1. Download and install [nmap](https://nmap.org/)
 
2. install [nmap](https://pypi.org/project/python-nmap/) python module
```
 pip install python-nmap
```
3. Set Workers and pss (parameter servers) devices name in [train_vgg19_distibuted](train_vgg19_distibuted.py)
 ```python
workers = ['PC1','PC2']
pss = ['PC3']
 ```
