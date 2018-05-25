"""
Simple tester for the vgg19_trainable
"""
import numpy as np
import tensorflow as tf

from dataSetGenerator import loadClasses
from dataSetGenerator import picShow
from vgg19 import vgg19_trainable as vgg19

#path= "C:/Users/{}/Desktop/UCMerced_LandUse/Images/".format(getlogin())
#batch, labels, classes = dataSetGenerator(path,True,224,80)
batch = np.load("DataSets/UCMerced_LandUse_DU_dataTrain.npy")
labels = np.load("DataSets/UCMerced_LandUse_DU_labelsTrain.npy")
classes = loadClasses("DataSets/UCMerced_LandUse.txt")
with tf.device('/device:GPU:0'):
# with tf.device('/cpu:0'):
    # with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(environ['NUMBER_OF_PROCESSORS']))) as sess:

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        true_out = tf.placeholder(tf.float32, [None, len(classes)])
        train_mode = tf.placeholder(tf.bool)

        vgg = vgg19.Vgg19('Weights/VGG19_21C.npy',len(classes))
        vgg.build(images,train_mode)

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        # print('number of variables used:',vgg.get_var_count())
        print('Data SHape used:',batch.shape)

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
        batche_num = len(batch)
        for _ in range(epochs):
                print("*******************  ", _, "  *******************")
                indice = np.random.permutation(batche_num)
                s = 0
                for i in range(int(batche_num/batch_size)):
                    min_batch = indice[i*batch_size:(i+1)*batch_size]
                    cur_cost, cur_train,cur_acc= sess.run([cost, train,acc], feed_dict={images: batch[min_batch], true_out: labels[min_batch], train_mode: True})
                    print("Iteration %d loss:\n%s" % (i, cur_cost))
                    with open('Data/cost19_21C.txt', 'a') as f:
                        f.write(str(cur_cost)+'\n')
                    with open('Data/acc19_21C.txt', 'a') as f:
                        f.write(str(cur_acc)+'\n')
                    s += 1
                    if s % 100 == 0:
                        # save Weights
                        vgg.save_npy(sess, 'Weights/VGG19_21C.npy')
                #  save Weights
                vgg.save_npy(sess, 'Weights/VGG19_21C.npy')

        # test classification again, should have a higher probability about tiger
        prob = sess.run(vgg.prob, feed_dict={images: batch[:10], train_mode: False})
        picShow(batch[:10],labels[:10], classes, None, prob)

        # import subprocess
        # subprocess.call(["shutdown", "/s"])
