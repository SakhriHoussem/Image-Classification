# Classification of Satellite images

## Classification of Satellite images :rocket: using VGG-Net and [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) for Training
### After Training : 
Resultat of the Model After Training of testing the classification one batch of Pictures [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) for Training the network
![afterTrain](images/afterTrain.png 'afterTrain')
#### `cost` and `accuracy` of training : 
![graph](images/graph.png 'graph')

#### Test a Picture : 
 Resultat after testing a Picture From Diffrent source **ex : [Google Map](maps.google.com)**
![testing](images/testing.png 'testing')
## How To use :

### Instalation :

* install [tensorflow 1.6](https://github.com/SakhriHoussem/How-to-install-tensorflow-gpu)
```python
 pip install tensorflow
 ```
* install python [matplotlib](https://matplotlib.org/)
```python
pip install matplotlib
```
* install python [opencv](https://pypi.org/project/opencv-python/)
```python
 pip install opencv-python
 ```
### Train the Model:
To Train Model for different DataSets or Different Classification follow the steps : 

1. Choose your images DataSet for Training

2. from [dataSetGenerator](dataSetGenerator.py) use `dataSetToNPY()` to Convert Your DataSet to `file.npy` for Dataset Fast Reading   
 
 ```python
dataSetToNPY(path,dataSet_name,resize=True,resize_to=224,percentage=80) 
 ```
3. the Output of `dataSetToNPY()` :      
`dataSet_name_dataTrain.npy` `dataSet_name_labelsTrain.npy`
`dataSet_name_dataTest.npy` `dataSet_name_labelsTest.npy` `dataSet_name_classes.npy`
   
4. in [train_vgg19](train_vgg19.py) or [train_vgg16](train_vgg16.py)     
   
   1. Get batch of images and Labels for training 
 ```python
 batch = np.load("dataSet_name_dataTrain.npy")
labels = np.load("dataSet_name_labelsTrain.npy")
```     

   2. get classes name
 ```python
 classes = np.load("dataSet_name_classes.npy")  # get classes name for file.txt
 ```
 or if you used `saveClasses(path,save_to,mode = 'w')` for generate Classes `dataSet_name_classes.txt`
 
```python
classes = loadClasses("dataSet_name_classes.txt")
``` 
   3. set `Weights.npy` if exist 
 ```python
 vgg = vgg19.Vgg19('Weights.npy')
 ```
 
   4. change `epochs` and `batch size` [optionnel]
```python
batch_size = 10
epochs = 30
```

   5. choose  path and file Name for each of `cost` and `accuracy`
```python
with open('Data/cost.txt', 'a') as f:
    f.write(str(cur_cost)+'\n')
with open('Data/acc.txt', 'a') as f:
    f.write(str(cur_acc)+'\n')
```

   6. choose  path and file Name for new `Weights`
```python
vgg.save_npy(sess, 'Weights/VGG19_21C.npy')
```

5. Change the number of Output Classes in [vgg19](vgg19/vgg19_trainable.py) or  [vgg16](vgg16/vgg16_trainable.py)
```python
self.fc8 = self.fc_layer(self.relu7, 4096, 21, "fc8") # change 21 to the number of classes you need
```
### For Distributed Tensorflow :

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
 
#### to Plot Graph of `cost` and `accuracy` :
 
from [dataSetGenerator](dataSetGenerator.py) use `plotFiles()`

```python
plotFiles(*path, xlabel='# epochs', ylabel='Error and Accu',autoClose = False)
```
![graph](images/graph.png 'graph')

### Test the Model :

from [test_vgg19](test_vgg19.py) or [test_vgg16](test_vgg16.py) :
#### read Image : 
to read One or Batch of pictures using 
```python
batch = imread(path) # read One or batch of pictures
```
get Classes name

 ```python
 classes = np.load("dataSet_name_classes.npy")  # get classes name for file.npy
 ```
or 
```python
classes = loadClasses("dataSet_name_classes.txt") # get classes name for file.txt
```
#### Show pictures : 
to Show one or batch of Pictures 
```python
picShow(data,labels,classes,just=None,predict=None,autoClose = False)
```
![afterTrain](images/afterTrain.png 'afterTrain')
