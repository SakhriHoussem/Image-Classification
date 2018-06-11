"""
Simple tester for the vgg16_trainable
"""
from os import environ
import numpy as np
import tensorflow as tf
from dataSetGenerator import picShow,append
from vgg16 import vgg16_trainable as vgg16

classes_name = "RSSCN7"
# classes_name = "UCMerced_LandUse_DU"
# classes_name = "SIRI-WHU"

classes = np.load("DataSets/{}_classes.npy".format(classes_name))
batch = np.load("DataSets/{}_dataTrain.npy".format(classes_name))
labels = np.load("DataSets/{}_labelsTrain.npy".format(classes_name))

classes_num = len(classes)
rib = batch.shape[1] # picture Rib
# with tf.device('/device:GPU:0'):
#     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
with tf.device('/cpu:0'):
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(environ['NUMBER_OF_PROCESSORS']))) as sess:


        images = tf.placeholder(tf.float32, [None, rib, rib, 3])
        true_out = tf.placeholder(tf.float32, [None, classes_num])
        train_mode = tf.placeholder(tf.bool)

        try:
            vgg = vgg16.Vgg16('Weights/VGG16_{}.npy'.format(classes_name),classes_num)
        except:
            print('Weights/VGG16_{}.npy Not Exist'.format(classes_name))
            vgg = vgg16.Vgg16(None,classes_num)
        vgg.build(images,train_mode)

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print('number of variables used:',vgg.get_var_count())

        sess.run(tf.global_variables_initializer())

        # test classification
        prob = sess.run(vgg.prob, feed_dict={images: batch[:10], train_mode: False})
        picShow(batch[:10],labels[:10], classes, None, prob,True)
        # simple 1-step training
        cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(prob), tf.argmax(true_out))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        batch_size = 10
        epochs = 30
        batche_num = batch.shape[0]
        costs = []
        accs = []
        for _ in range(epochs):
                indice = np.random.permutation(batche_num)
                counter = 0
                for i in range(int(batche_num/batch_size)):
                    min_batch = indice[i*batch_size:(i+1)*batch_size]
                    cur_cost, cur_train,cur_acc= sess.run([cost, train,acc], feed_dict={images: batch[min_batch], true_out: labels[min_batch], train_mode: True})
                    print("Iteration :{} Batch :{} loss :{}".format(_, i, cur_cost))
                    accs.append(cur_acc)
                    costs.append(cur_cost)
                    counter += 1
                    if counter % 100 == 0:
                        #  save graph data
                        append(costs,'Data/COST16_{}.txt'.format(classes_name))
                        append(accs,'Data/ACC16_{}.txt'.format(classes_name))
                        # save Weights
                        vgg.save_npy(sess, 'Weights/VGG16_{}.npy'.format(classes_name))

                #  save graph data
                append(costs,'Data/COST16_{}.txt'.format(classes_name))
                append(accs,'Data/ACC16_{}.txt'.format(classes_name))
                #  save Weights
                vgg.save_npy(sess, 'Weights/VGG16_{}.npy'.format(classes_name))
        # test classification again, should have a higher probability about tiger
        prob = sess.run(vgg.prob, feed_dict={images: batch[:10], train_mode: False})
        picShow(batch[:10],labels[:10], classes,None,prob)

        # import subprocess
        # subprocess.call(["shutdown", "/s"])
