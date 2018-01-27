"""
@author: Arjun Krishna
@desc: A sample tf-Model
"""
import tensorflow as tf

class Model:

  def __init__(self, inputs, target):
    input_size = int(inputs.get_shape()[1])
    target_size = int(target.get_shape()[1])

    with tf.variable_scope('network'):
      W = tf.get_variable(
            "W", 
            shape=[input_size, target_size],
            initializer=tf.contrib.layers.xavier_initializer())
      b = tf.get_variable(
            "b",
            shape=[target_size],
            initializer=tf.zeros_initializer())

      output = tf.matmul(inputs, W) + b
      self._prediction = tf.nn.softmax(output, name="prediction")
    
    with tf.name_scope('loss'):
      self._loss = -tf.reduce_sum(target*tf.log(self._prediction))
      
    self._optimize = tf.train.AdamOptimizer(1e-1).minimize(self._loss)
    
    with tf.name_scope('error'):
      mistakes = tf.not_equal(
          tf.argmax(target, 1), tf.argmax(self._prediction, 1))
      self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

  @property
  def prediction(self):
    return self._prediction

  @property
  def optimize(self):
    return self._optimize

  @property
  def loss(self):
    return self._loss

  @property
  def error(self):
    return self._error
