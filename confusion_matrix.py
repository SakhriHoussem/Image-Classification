"""
Draw Confusion Matrix for the vgg19
"""
import tensorflow as tf
import vgg19.vgg19 as vgg19
from dataSetGenerator import picShow
from dataSetGenerator import dataSetGenerator
from dataSetGenerator import confusion_matrix
from dataSetGenerator import draw_confusion_matrix
from dataSetGenerator import draw_table
from os import getlogin
import numpy as np
batch,labels,classes = dataSetGenerator("C:\\Users\\shous\Desktop\datasets\\UCMerced_LandUse\\Images")
conf_mat = np.zeros((len(classes),len(classes)))
batch_size = 10
batche_num = len(batch)
classes_num = len(classes)
rib = batch.shape[1] # picture Rib
indice = np.random.permutation(batche_num)
with tf.device('/device:cpu:0'):
# with tf.device('/device:GPU:0'):
    # with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(environ['NUMBER_OF_PROCESSORS']))) as sess:
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        images = tf.placeholder(tf.float32, [None, rib, rib, 3])
        vgg = vgg19.Vgg19("Weights/VGG19_"+str(classes_num)+"C.npy") #set the path
        with tf.name_scope("content_vgg"):
                vgg.build(images)
        for i in range(int(batche_num/batch_size)):
            min_batch = indice[i*batch_size:(i+1)*batch_size]
            tru = labels[min_batch]
            prob = sess.run(vgg.prob,{images: batch[min_batch]})
            conf_mat += confusion_matrix(prob,tru,classes)
            print("Iteration %d" % i)
        draw_confusion_matrix(conf_mat,classes)
        draw_table(conf_mat,classes)
        np.save('Data/confusion_matrix.npy',conf_mat)

        picShow(batch, labels, classes, None, prob)