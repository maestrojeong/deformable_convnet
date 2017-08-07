from ops import mnistloader, conv2d, deform_conv2d, print_vars
from configs import DeformConvConfig, SAVE_DIR
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import logging
import os

logging.basicConfig(format = "[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DeformConv(DeformConvConfig):
    def __init__(self, dataset = 'MNIST'):
        logger.info("Initialization begins")
        DeformConvConfig.__init__(self)
        logger.info("MNIST dataset load begins")
        self.train, self.test, self.val = mnistloader("../MNIST_data/")
        logger.info("MNIST dataset load done...")

        self.image = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 784])
        self.label = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 10])
	self.label_logit = self.build_model(self.image)
	
        self.sess=tf.Session()	
        self.cross_entropy = softmax_cross_entropy(logits=self.label_logit, labels= self.label)
        self.run_train = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
	self.accuracy = tf.reduce_mean(tf.argmax(self.label, axis=1), tf.argmax(self.label_logit, axis=1))
 
    def build_model(self, x):
        logger.info("Buidling model starts...")
        o = deform_conv2d(x, [7,7,1,50], [5,5,1,32], activation=tf.nn.relu, scope="deform_conv1") 
        o = tf.nn.max_pool(o, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        o = deform_conv2d(x, [7,7,32,50], [5,5,32,64], activation=tf.nn.relu, scope="deform_conv1") 
        o = tf.nn.max_pool(o, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        o = tf.reshape(o, [self.batch_size, -1])
        o = fc_layer(o, 512, activation=tf.nn.relu, scope="fc1")
	o = fc_layer(o, 10, scope="fc2") 
	print_vars("trainable_variables")
        logger.info("Buidling model done")
	return o
    
    def restore(self):
        logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(SAVE_DIR))
        logger.info("Restoring model done.")

    def train(self):
	create_dir(SAVE_DIR)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 10)
        ntrain = len(self.train.image)
        nbatch = ntrain//self.batch_size

        for epoch in range(self.epoch):
            # shuffle start
            index = np.arange(length)
            np.random.shuffle(index)
            shuffle_image = self.train.images[index]
            shuffle_label = self.train.labels[index]
            # shuffle end
	    epoch_accuracy = 0
            for batch in tqdm(range(self.nbatch), ascii = True, desc = "batch"):
		train_feed_dict = {self.image : shuffle_image[self.batch_size*batch:self.batch_size*(batch+1)], self.label : shuffle_label[self.batch_size*batch:self.batch_size*(batch+1)}
                _, batch_accuracy = self.sess.run([self.run_train, self.accuracy], feed_dict = train_feed_dict)
		epoch_accuracy += batch_accuracy

	    epoch_accuracy/=self.nbatch
            if epoch%self.log_every == self.log_every-1:
		test_feed_dict = {self.image : self.test.images, self.label : self.test.labels}
		test_accuracy = self.sess.run(self.accuracy, feed_dict=test_feed_dict)
                print("Epoch({}/{}) train_accuracy : {}%, test_accuracy : {}%".format(epoch+1, self.epoch, epoch_accuracy, test_accuracy))
            saver.save(sess, os.path.join(SAVE_DIR, 'model'), global_step = epoch+1)

if __name__ = '__main__':
    model = DeformConv()
    model.train()
