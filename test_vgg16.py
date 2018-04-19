"""
Simple tester for the vgg16
"""
import tensorflow as tf
import vgg16.vgg16 as vgg16
from dataSetGenerator import picShow
from dataSetGenerator import imread
from dataSetGenerator import getLabel
from dataSetGenerator import loadClasses
from os import getlogin
from os import environ

path = "C:/Users/{}/Desktop/UCMerced_LandUse/Images/golfcourse/golfcourse84.tif".format(getlogin())# picture path
classe = path.split("/")[-2] # get classe name from path

classes = loadClasses("Weights/UCMerced_LandUse.txt") # get classes name
batch = imread(path) # read one picture
label = getLabel(classe,classes)  # get the right label for the picture

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(environ['NUMBER_OF_PROCESSORS']))) as sess:
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])

        vgg = vgg16.Vgg16("Weights/VGG16_21C.npy") #set the path
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, {images: batch})
        picShow(batch, label, classes, None, prob)