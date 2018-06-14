"""
Simple tester for the vgg19
"""
import tensorflow as tf
import vgg19.vgg19 as vgg19
from dataSetGenerator import picShow
from numpy import load
from os import environ
import argparse

parser = argparse.ArgumentParser(prog="Test vgg19",description="tester for the vgg19_trainable")
parser.add_argument('--dataset', metavar='dataset', type=str,required=True,
                    help='DataSet Name')
parser.add_argument('--batch', metavar='batch', type=int, default=12, help='batch size ')

args = parser.parse_args()


classes_name = args.dataset
batch_size = args.batch
# batch_size = 12
# classes_name = "RSSCN7"
# classes_name = "SIRI-WHU"
# classes_name = "UCMerced_LandUse"

classes = load("DataSets/{0}/{0}_classes.npy".format(classes_name))
batch = load("DataSets/{0}/{0}_dataTest.npy".format(classes_name)) # read one picture
label =load("DataSets/{0}/{0}_labelsTest.npy".format(classes_name))
rib = batch.shape[1]

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
