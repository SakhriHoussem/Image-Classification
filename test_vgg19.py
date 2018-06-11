"""
Simple tester for the vgg19
"""
import tensorflow as tf
import vgg19.vgg19 as vgg19
from dataSetGenerator import picShow
from numpy import load
from os import environ
classes_name = "RSSCN7"
# classes_name = "SIRI-WHU"
# classes_name = "UCMerced_LandUse"

classes = load("DataSets/{0}/{0}_classes.npy".format(classes_name))
batch = load("DataSets/{0}/{0}_dataTest.npy".format(classes_name)) # read one picture
label =load("DataSets/{0}/{0}_labelsTest.npy".format(classes_name))
rib = batch.shape[1]

batch_size = 12
with tf.device('/device:cpu:0'):
# with tf.device('/device:GPU:0'):
#     with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(environ['NUMBER_OF_PROCESSORS']))) as sess:
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        images = tf.placeholder(tf.float32, [None, rib, rib, 3])
        vgg = vgg19.Vgg19("Weights/VGG19_{}.npy".format(classes_name)) #set the path
        with tf.name_scope("content_vgg"):
                vgg.build(images)
        prob = sess.run(vgg.prob, {images: batch[:batch_size]})
        picShow(batch[:batch_size], label[:batch_size], classes, None, prob,Save_as="test19_{}".format(classes_name))
