"""
Simple tester for the vgg19_trainable
"""
import numpy as np
import tensorflow as tf
from dataSetGenerator import append
from dataSetGenerator import picShow
from vgg19 import vgg19_trainable as vgg19

# classes_name = "SIRI-WHU"
# classes_name = "UCMerced_LandUse"
classes_name = "RSSCN7"

classes = np.load("DataSets/{0}/{0}_classes.npy".format(classes_name))
batch = np.load("DataSets/{0}/{0}_dataTrain.npy".format(classes_name))
labels = np.load("DataSets/{0}/{0}_labelsTrain.npy".format(classes_name))

classes_num = len(classes)

rib = batch.shape[1]
with tf.device('/device:GPU:0'):
# with tf.device('/cpu:0'):
    # with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(environ['NUMBER_OF_PROCESSORS']))) as sess:

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

        images = tf.placeholder(tf.float32, [None, rib, rib, 3])
        true_out = tf.placeholder(tf.float32, [None, classes_num])
        train_mode = tf.placeholder(tf.bool)
        try:
            vgg = vgg19.Vgg19('Weights/VGG19_{}.npy'.format(classes_name),classes_num)
        except:
            print('Weights/VGG19_{}.npy Not Exist'.format(classes_name))
            vgg = vgg19.Vgg19(None,classes_num)
        vgg.build(images,train_mode)

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        # print('number of variables used:',vgg.get_var_count())
        print('Data SHape used:',batch.shape)

        sess.run(tf.global_variables_initializer())

        # test classification
        prob = sess.run(vgg.prob, feed_dict={images: batch[:8], train_mode: False})
        # picShow(batch[:8], labels[:8], classes, None, prob, True)

        # simple 1-step training
        cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(prob), tf.argmax(true_out))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        batch_size = 10
        epochs = 30
        batche_num = len(batch)
        accs = []
        costs = []
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
                        # save Weights
                        vgg.save_npy(sess, 'Weights/VGG19_{}.npy'.format(classes_name))
                        # save graph data
                        append(costs,'Data/cost19_{}.txt'.format(classes_name))
                        append(accs,'Data/acc19_{}.txt'.format(classes_name))
                #  save Weights
                vgg.save_npy(sess, 'Weights/VGG19_{}.npy'.format(classes_name))


        # test classification again, should have a higher probability about tiger
        prob = sess.run(vgg.prob, feed_dict={images: batch[:8], train_mode: False})
        picShow(batch[:8], labels[:8], classes, None, prob)

        # import subprocess
        # subprocess.call(["shutdown", "/s"])
