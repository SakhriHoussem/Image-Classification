"""
Simple tester for the vgg19
"""
import tensorflow as tf
import vgg19.vgg19 as vgg19
from dataSetGenerator import picShow
from dataSetGenerator import imread
from dataSetGenerator import getLabel
from dataSetGenerator import loadClasses
from os import getlogin

path = "C:/Users/{}/Desktop/datasets/UCMerced_LandUse/Images/mediumresidential/mediumresidential84.tif".format(getlogin())# picture path
classe = path.split("/")[-2] # get classe name from path
classes = loadClasses("Datasets/UCMerced_LandUse.txt") # get classes name
batch = imread(path) # read one picture
label = getLabel(classe,classes)  # get the right label for the picture

with tf.device('/device:cpu:0'):
# with tf.device('/device:GPU:0'):
    # with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(environ['NUMBER_OF_PROCESSORS']))) as sess:
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg19.Vgg19("Weights/VGG19_21C.npy") #set the path
        with tf.name_scope("content_vgg"):
                vgg.build(images)
        prob = sess.run(vgg.prob, {images: batch})
        picShow(batch, label, classes, None, prob)
