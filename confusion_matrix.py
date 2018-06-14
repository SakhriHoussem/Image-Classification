"""
Draw Confusion Matrix for the vgg19
"""
import tensorflow as tf
import vgg19.vgg19 as vgg19
from dataSetGenerator import dataSetGenerator, picShow,draw_table, confusion_matrix, draw_confusion_matrix
from os import getlogin
import numpy as np
import argparse

parser = argparse.ArgumentParser(prog="Draw Confusion Matrix",description="Draw Confusion Matrix for the vgg19")
parser.add_argument('--dataset', metavar='dataset', type=str,required=True,
                    help='DataSet Name')
parser.add_argument('--batch', metavar='batch', type=int, default=12, help='batch size ')
parser.add_argument('--showPic', metavar='showPic', type=bool, default=False, help='Show patch of picture each epoch')

args = parser.parse_args()


classes_name = args.dataset
batch_size = args.batch

# classes_name = "UCMerced_LandUse"
# classes_name = "SIRI-WHU"
# classes_name = "RSSCN7"
# batch_size = 10

classes = np.load("DataSets/{0}/{0}_classes.npy".format(classes_name))
batch = np.load("DataSets/{0}/{0}_dataTest.npy".format(classes_name)) # read one picture
labels =np.load("DataSets/{0}/{0}_labelsTest.npy".format(classes_name))

# batch,labels,classes = dataSetGenerator("C:\\Users\\{}\Desktop\datasets\\{}\\".format(getlogin(),classes_name),resize_to=400)


conf_mat = np.zeros((len(classes),len(classes)))

batche_num = len(batch)
classes_num = len(classes)

rib = batch.shape[1] # picture Rib
indice = np.random.permutation(batche_num)
with tf.device('/device:cpu:0'):
# with tf.device('/device:GPU:0'):
    # with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(environ['NUMBER_OF_PROCESSORS']))) as sess:
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        images = tf.placeholder(tf.float32, [None, rib, rib, 3])
        vgg = vgg19.Vgg19("Weights/VGG19_{}.npy".format(classes_name)) #set the path
        with tf.name_scope("content_vgg"):
            vgg.build(images)
        for i in range(int(batche_num/batch_size)):
            min_batch = indice[i*batch_size:(i+1)*batch_size]
            tru = labels[min_batch]
            prob = sess.run(vgg.prob,{images: batch[min_batch]})
            if args.showPic: picShow(batch[:min_batch], labels[:min_batch], classes, None, prob)
            conf_mat += confusion_matrix(prob,tru,classes)
            print("batch :{}".format(i))
        np.save('Data/cf_mat_{}.npy'.format(classes_name),conf_mat)
        draw_confusion_matrix(conf_mat,classes,save_as="cf_mat_{}".format(classes_name))
        draw_table(conf_mat,classes, save_as="table_precision_{}".format(classes_name))
