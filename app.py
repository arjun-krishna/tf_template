"""
@author: Arjun Krishna
@desc: tf-application
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import os
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'Mode of application : train (or) sample')
flags.DEFINE_integer('max_steps', 10, 'Numer of epochs')
flags.DEFINE_string('log_path', 'log_dir', 'Directory to log summaries')
flags.DEFINE_string('save_path', 'chkpt', 'Directory to save model')

FLAGS = flags.FLAGS

from model import Model
from data import DataManager

dm = DataManager()

inputs = tf.placeholder(tf.float32, shape=[None, 20], name="inputs")
target = tf.placeholder(tf.float32, shape=[None, 10], name="target")

with tf.variable_scope('Model'):
  model = Model(inputs, target)

tf.summary.scalar('error', model.error)
tf.summary.scalar('loss', model.loss)

merged = tf.summary.merge_all()

def main(_):
  
  saver = tf.train.Saver()
  
  FLAGS.mode = 1 if FLAGS.mode == 'train' else 0

  with tf.Session() as sess:

    if FLAGS.mode:
      summary_writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
      tf.global_variables_initializer().run()
      
      pbar = tqdm(total=FLAGS.max_steps)
      for i in range(FLAGS.max_steps):
        bX, by = dm.get_batch()
        summary, _ = sess.run([merged, model.optimize], feed_dict={inputs: bX, target: by})
        summary_writer.add_summary(summary, i)
        pbar.update(1)
      pbar.close()

    else:
    	saver.restore(sess, os.path.join(FLAGS.save_path, "model.ckpt"))

    # simply test (works with train and sample mode)
    print (sess.run(model.prediction, feed_dict={inputs: np.zeros((1,20))}))  

    if FLAGS.mode:
    	save_path = saver.save(sess, os.path.join(FLAGS.save_path, "model.ckpt"))
    	print ('Model saved in %s' % save_path)

if __name__ == "__main__":
  tf.app.run()