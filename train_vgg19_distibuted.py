"""
Simple tester for the Tensorflow distributed
"""
from os import environ
import numpy as np
import tensorflow as tf
from Networking import ClusterGen
from dataSetGenerator import append
from dataSetGenerator import loadClasses
from dataSetGenerator import picShow
from vgg19 import vgg19_trainable as vgg19

batch = np.load("DataSets/UCMerced_LandUse_dataTrain.npy")
labels = np.load("DataSets/UCMerced_LandUse_labelsTrain.npy")
classes = loadClasses("DataSets/UCMerced_LandUse.txt")
classes_num = len(classes)
rib = batch.shape[1] # picture Rib

workers = ['DESKTOP-07HFBQN','FOUZI-PC']
pss = ['DELL-MINI']

index = workers.index(environ['COMPUTERNAME']) if environ['COMPUTERNAME'] in workers else pss.index(environ['COMPUTERNAME']) if environ['COMPUTERNAME'] in pss else None
job = 'worker' if environ['COMPUTERNAME'] in workers else 'ps' if environ['COMPUTERNAME'] in pss else None

cluster = tf.train.ClusterSpec(ClusterGen(workers,pss))
server = tf.train.Server(cluster, job_name=job, task_index=index)

# with tf.device(tf.train.replica_device_setter(
#                    worker_device="/job:ps/task:"+str(index),
#                    cluster=cluster)):
with tf.device("/job:ps/task:"+str(index)):

    images = tf.placeholder(tf.float32, [None, rib, rib, 3])
    true_out = tf.placeholder(tf.float32, [None, len(classes)])
    train_mode = tf.placeholder(tf.bool)
    try:
        vgg = vgg19.Vgg19('Weights/VGG19_'+str(classes_num)+'C.npy',len(classes))
    except:
        vgg = vgg19.Vgg19(None,len(classes))

    vgg.build(images,train_mode)

    global_step = tf.train.get_or_create_global_step()
    inc_global_step = tf.assign(global_step, global_step + 1)

if job == 'worker':
    # with tf.device(tf.train.replica_device_setter(
    #                    worker_device="/job:worker/task:"+str(index),
    #                    cluster=cluster)):
    with tf.device("/job:worker/task:"+str(index)):

        cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(cost), tf.argmax(true_out))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        batch_size = 5
        batche_num = len(batch)
        # tf.train.global_step(sess, tf.Variable(10, trainable=False, name='global_step'))
        hooks=[tf.train.StopAtStepHook(last_step=2)]
    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(index == 0), hooks=hooks) as sess:
        while not sess.should_stop():
            _ = sess.run(inc_global_step)
            costs = []
            accs = []
            print("*******************  ", _, "  *******************")
            indice = np.random.permutation(batche_num)
            for i in range(int(batche_num/batch_size)):
                print('step 1')
                min_batch = indice[i*batch_size:(i+1)*batch_size]
                print('step 2')
                cur_cost, cur_train,cur_acc= sess.run([cost, train,acc], feed_dict={images: batch[min_batch], true_out: labels[min_batch], train_mode: True})
                print("Iteration %d loss:\n%s" % (i, cur_cost))
                costs.append(str(cur_cost)+'\n')
                accs.append(str(cur_acc)+'\n')
            # with tf.device(tf.train.replica_device_setter(
            #                    worker_device="/job:ps/task:"+str(index),
            #                    cluster=cluster)):
            append(costs,'Data/cost19_'+str(classes_num)+'C_D')
            append(accs,'Data/acc19_'+str(classes_num)+'C_D')
            vgg.save_npy(sess, 'Weights/VGG19_'+str(classes_num)+'C_D.npy')
        # test classification
        prob = sess.run(vgg.prob, feed_dict={images: batch[:10], train_mode: False})
        picShow(batch[:10],labels[:10], classes, None, prob)
elif job == 'ps':
    server.join()
else:
    print("error JOB")

