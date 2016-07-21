import tensorflow as tf
import numpy as np
from ale_python_interface import ALEInterface

class Network(object):
    @staticmethod
    def weight_var(shape, **kwargs):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), **kwargs)

    @staticmethod
    def bias_var(shape, **kwargs):
        return tf.Variable(tf.constant(0.1, shape=shape), **kwargs)

    @staticmethod
    def conv(input, weights, stride, **kwargs):
        return tf.nn.conv2d(
                input,
                weights,
                strides=[1, stride, stride, 1],
                padding="SAME",
                **kwargs
            )

    def __init__(self, num_actions):
        self.num_actions = num_actions

        print("num_actions = {}".format(num_actions))

        # Input layer
        image     = tf.placeholder("float", [None, 84, 84, 4])

        # First convolutional layer
        with tf.variable_scope("conv1"):
            weights_1 = self.weight_var([8, 8, 4, 16], name="weight")
            bias_1    = self.bias_var([16], name="bias")

            _ = tf.histogram_summary('conv1_weights', weights_1)
            _ = tf.histogram_summary('conv1_bias', bias_1)

            conv_1    = self.conv(image, weights_1, 4, name="conv")
            biased_1  = conv_1 + bias_1
            relu_1    = tf.nn.relu(biased_1, "relu")

            _ = tf.histogram_summary('conv1_conv', conv_1)
            _ = tf.histogram_summary('conv1_biased', biased_1)
            _ = tf.histogram_summary('conv1_relu', relu_1)

        # Second convolutional layer
        with tf.variable_scope("conv2"):
            weights_2 = self.weight_var([4, 4, 16, 32], name="weight")
            bias_2    = self.bias_var([32], name="bias")

            _ = tf.histogram_summary('conv2_weights', weights_2)
            _ = tf.histogram_summary('conv2_bias', bias_2)

            conv_2    = self.conv(relu_1, weights_2, 2, name="conv")
            biased_2  = conv_2 + bias_2
            relu_2    = tf.nn.relu(biased_2, "relu")

            _ = tf.histogram_summary('conv2_conv', conv_2)
            _ = tf.histogram_summary('conv2_biased', biased_2)
            _ = tf.histogram_summary('conv2_relu', relu_2)

            reshape_2 = tf.reshape(relu_2, [-1, 3872])

        # Fully connected layer
        with tf.variable_scope("fully_conn"):
            weights_fc = self.weight_var([3872, 256], name="weight")
            bias_fc    = self.bias_var([256], name="bias")

            _ = tf.histogram_summary('fc_weights', weights_fc)
            _ = tf.histogram_summary('fc_bias', bias_fc)

            fc        = tf.matmul(reshape_2, weights_fc, name="multiply")
            biased_fc = fc + bias_fc
            relu_fc   = tf.nn.relu(biased_fc)

            _ = tf.histogram_summary('fc_matmul', fc)
            _ = tf.histogram_summary('fc_biased', biased_fc)
            _ = tf.histogram_summary('fc_relu', relu_fc)

        # Readout layer
        with tf.variable_scope("readout"):
            weights_r = self.weight_var([256, num_actions], name="weight")
            bias_r    = self.bias_var([num_actions], name="bias")

            _ = tf.histogram_summary('readout_weights', weights_r)
            _ = tf.histogram_summary('readout_bias', bias_r)

            readout   = tf.matmul(relu_fc, weights_r)
            biased_r  = readout + bias_r

            _ = tf.histogram_summary('readout_matmul', readout)
            _ = tf.histogram_summary('readout_biased', biased_r)

        self.image = image
        _ = tf.histogram_summary('image', image)

        self.readout = biased_r

        with tf.variable_scope("result"):
            self.actions   = tf.placeholder("float", [None, num_actions], name="action_mask")
            self.rewards   = tf.placeholder("float", [None, 1], name="rewards")

            self.predicted_rewards = tf.reduce_sum(
                    tf.mul(self.readout, self.actions, name="multiply"),
                    reduction_indices = [1,],
                    name="reduce"
                )

            _ = tf.histogram_summary('predictions', self.predicted_rewards)

            self.cost = tf.reduce_mean(tf.square(self.rewards - self.predicted_rewards), name="cost")

            _ = tf.scalar_summary('cost', self.cost)
