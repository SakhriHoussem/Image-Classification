# """
# Simple tester for the vgg19_trainable
# """
# import numpy as np
# import tensorflow as tf
#
# from dataSetGenerator import dataSetGenerator
# from dataSetGenerator import picShow
# from dataSetGenerator import confusion_matrix
# from dataSetGenerator import draw_confusion_matrix
# from dataSetGenerator import draw_table
# from vgg19 import vgg19_trainable as vgg19
#
# batch,labels,classes = dataSetGenerator("C:\\Users\\shous\Desktop\datasets\\UCMerced_LandUse\\Images")
#
# # with tf.device('/device:GPU:0'):
# # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
#
# with tf.device('/cpu:0'):
#         with tf.Session() as sess:
#
#                 images = tf.placeholder(tf.float32, [None, 224, 224, 3])
#                 true_out = tf.placeholder(tf.float32, [None, len(classes)])
#                 train_mode = tf.placeholder(tf.bool)
#
#                 vgg = vgg19.Vgg19('Weights/VGG19_21C.npy',len(classes))
#                 vgg.build(images,train_mode)
#
#                 # print number of variables used: 139754605 variables, i.e. ideal size = 548MB
#                 # print('number of variables used:',vgg.get_var_count())
#
#                 sess.run(tf.global_variables_initializer())
#
#                 # test classification
#                 prob = sess.run(vgg.prob, feed_dict={images: batch[:20], train_mode: False})
#                 picShow(batch[:10],labels[:10], classes, None, prob)
#
#                 # simple 1-step training
#                 cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
#                 train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
#
#                 correct_prediction = tf.equal(tf.argmax(prob), tf.argmax(true_out))
#                 acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#                 conf_mat = np.zeros((len(classes),len(classes)))
#                 batch_size = 20
#                 batche_num = len(batch)
#                 indice = np.random.permutation(batche_num)
#                 for i in range(int(batche_num/batch_size)):
#                     min_batch = indice[i*batch_size:(i+1)*batch_size]
#                     prob = sess.run(vgg.prob, feed_dict={images: batch[min_batch], true_out: labels[min_batch], train_mode: True})
#                     tru = labels[min_batch]
#                     conf_mat += confusion_matrix(prob,tru,classes)
#                     print("Iteration %d" % i)
#                 draw_confusion_matrix(conf_mat,classes)
#                 draw_table(conf_mat,classes)
#                 np.save('Data/confusion_matrix.npy',conf_mat)
#
#                 # test save
#                 vgg.save_npy(sess, 'Weights/VGG19_21C.npy')
#
#         # test classification again, should have a higher probability about tiger
#         # prob = sess.run(vgg.prob, feed_dict={images: batch[:10], train_mode: False})
#         # picShow(batch[:10],labels[:10], classes, None, prob)

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
indice = np.random.permutation(batche_num)
with tf.device('/device:cpu:0'):
# with tf.device('/device:GPU:0'):
    # with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(environ['NUMBER_OF_PROCESSORS']))) as sess:
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg19.Vgg19("Weights/VGG19_21C.npy") #set the path
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