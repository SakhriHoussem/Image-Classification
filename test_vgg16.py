"""
Simple tester for the vgg16
"""
import tensorflow as tf
import vgg16.vgg16 as vgg16
from dataSetGenerator import picShow
from numpy import load
from os import environ

classes_name = "RSSCN7"
classes_name = "SIRI-WHU"
classes_name = "UCMerced_LandUse"

classes = load("DataSets/{0}/{0}_classes.npy".format(classes_name))
batch = load("DataSets/{0}/{0}_dataTest.npy".format(classes_name)) # read one picture
label =load("DataSets/{0}/{0}_labelsTest.npy".format(classes_name))

rib = batch.shape[1]

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(environ['NUMBER_OF_PROCESSORS']))) as sess:
        images = tf.placeholder(tf.float32, [None, rib, rib, 3])

        vgg = vgg16.Vgg16("Weights/VGG16_{}.npy".format(classes_name)) #set the path
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, {images: batch})
        picShow(batch, label, classes, None, prob, Save_as="test19_{}.png".format(classes_name))