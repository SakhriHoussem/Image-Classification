# Classification of Satellite images

## Classification of Satellite images :rocket: using VGG-Net and [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) for Training the network
### Resultat After Trainning : 

![afterTrain](images/afterTrain.png 'afterTrain')
### `cost` and `accuracy` of training : 
![graph](images/graph.png 'graph')

### Resultat of testing a Picture : 

![testing](images/testing.png 'testing')
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
### How To Train Model for different DataSets or Different Classification :

* Choose your images DataSet for Training

* from [dataSetGenerator](dataSetGenerator.py) use `dataSetToNPY()` to Convert Your DataSet to `file.npy` for Dataset Fast Reading   
 
 ```python
dataSetToNPY(path,dataSet_name,resize=True,resize_to=224,percentage=80) 
 ```
* the Output of `dataSetToNPY()` :      
`dataSet_name_dataTrain.npy` `dataSet_name_labelsTrain.npy`
`dataSet_name_dataTest.npy` `dataSet_name_labelsTest.npy` `dataSet_name_classes.npy`
   
* in [train_vgg19](train_vgg19.py) or [train_vgg16](train_vgg16.py)     
   
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
 
 4. change `epochs` and `batch size` if you want
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

- Change the number of Output Classes in [vgg19](vgg19/vgg19_trainable.py) or  [vgg16](vgg16/vgg16_trainable.py)
```python
self.fc8 = self.fc_layer(self.relu7, 4096, 21, "fc8") # change 21 to the number of classes you need
```
### For Distributed Tensorflow :

* Download and install [nmap](https://nmap.org/)
 
* install [nmap](https://pypi.org/project/python-nmap/) python module
```
 pip install python-nmap
```
* Set Workers and pss (parameter servers) devices name in [train_vgg19_distibuted](train_vgg19_distibuted.py)
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

in [test_vgg19](test_vgg19.py) or [test_vgg16](test_vgg16.py) :
```python
batch = imread(path) # read One or lot of picture
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
```python
picShow(data,labels,classes,just=None,predict=None,autoClose = False)
```
![afterTrain](images/afterTrain.png 'afterTrain')
