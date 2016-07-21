import DataSet
import numpy as np
import tensorflow as tf


from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops

class Model(object):
    @staticmethod
    def weight_var(shape, gain, **kwargs):
        if len(shape) == 4:
            fan_in = np.prod(shape[:3])
        elif len(shape) == 2:
            fan_in = shape[0]
        else:
            raise RuntimeError("Invalid shape for Glorot")
        bound = np.sqrt(gain / fan_in)
        return tf.Variable(tf.truncated_normal(shape, stddev=bound), **kwargs)

    @staticmethod
    def bias_var(shape, value=0.1, **kwargs):
        return tf.Variable(tf.constant(value, shape=shape), **kwargs)

    @staticmethod
    def conv(input_, weights, stride, **kwargs):
        return tf.nn.conv2d(
                input_,
                weights,
                strides=[1, stride, stride, 1],
                padding="SAME",
                **kwargs
            )

    @staticmethod
    def max_pool(input_, ksize, stride, **kwargs):
        return tf.nn.max_pool(
                input_,
                ksize=[1, ksize, ksize, 1],
                strides=[1, stride, stride, 1],
                padding="SAME",
                **kwargs
            )

    def conv_layer(self, input_, in_channels, out_channels, filter_size, stride,
            name, nonlinearity=tf.nn.relu):
        with tf.variable_scope(name):
            weights = self.weight_var(
                    [filter_size, filter_size, in_channels, out_channels],
                    gain=2.0,
                    name="weights"
                )

            bias = self.bias_var(
                    [out_channels],
                    name="bias"
                )

            conv = self.conv(input_, weights, stride, name="conv")
            biased = conv + bias
            normalised = self.batch_norm(biased, name="batch_norm", over=[0, 1, 2])
            nonlinear = nonlinearity(normalised)

            tf.histogram_summary(name + "/conv", conv)
            tf.histogram_summary(name + "/biased", biased)
            tf.histogram_summary(name + "/normalised", normalised)
            tf.histogram_summary(name + "/nonlinearity", nonlinear)

            return nonlinear

    def pool_layer(self, input_, ksize, stride, name):
        with tf.variable_scope(name):
            pooled = self.max_pool(input_, ksize, stride, name="name")

            tf.histogram_summary(name + "/pooled", pooled)

            return pooled

    def fc_layer(self, input_, in_dimension, out_dimension, name, nonlinearity=tf.nn.relu):
        with tf.variable_scope(name):
            weights = self.weight_var(
                    [in_dimension, out_dimension],
                    gain=2.0,
                    name="weights"
                )

            bias = self.bias_var(
                    [out_dimension],
                    name="bias"
                )

            reshaped = tf.reshape(input_, [-1, in_dimension])
            matmul = tf.matmul(reshaped, weights)
            biased = matmul + bias
            normalised = self.batch_norm(biased, name="batch_norm", over=[0])
            nonlinear = nonlinearity(normalised)

            tf.histogram_summary(name + "/matmul", matmul)
            tf.histogram_summary(name + "/biased", biased)
            tf.histogram_summary(name + "/normalised", normalised)
            tf.histogram_summary(name + "/nonlinearity", nonlinear)

            return nonlinear

    def dropout(self, input_, keep_prob):
        with ops.op_scope([input_], None, "dropout") as name:
            rands = keep_prob + random_ops.random_uniform(
                array_ops.shape(input_))
            floored = math_ops.floor(rands)
            ret = input_ * math_ops.inv(keep_prob) * floored
            ret.set_shape(input_.get_shape())
            return ret

    def batch_norm(self, input_, name, over=[0]):
        with tf.variable_scope(name):
            batch_mean, batch_var = tf.nn.moments(
                    input_,
                    over,
                    name='moments',
                )

            batch_inv_std = 1/(tf.sqrt(batch_var) + 1e-6)

            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            ema_op = ema.apply([batch_mean, batch_inv_std])

            ema_mean = ema.average(batch_mean)
            ema_inv_std = ema.average(batch_inv_std)

            def with_updates():
                with tf.control_dependencies([ema_op]):
                    return tf.identity(batch_mean), tf.identity(batch_inv_std)

            def without_updates():
                return (ema_mean, ema_inv_std)

            mean, inv_std = control_flow_ops.cond(
                    self.use_precomputed_means,
                    without_updates,
                    with_updates,
                )

            #tf.scalar_summary('ema/mean', ema_mean)
            #tf.scalar_summary('ema/inv_std', ema_inv_std)

            return (input_ - mean) * inv_std


    def __init__(self):
        self.input_var = tf.placeholder(
                'float',
                [None, DataSet.WIDTH, DataSet.HEIGHT, DataSet.NUM_CHANNELS],
                name="input_var",
            )

        self.corr_labels = tf.placeholder(
                'float',
                [None, 10],
                name="corr_labels",
            )

        self.keep_prob = tf.placeholder(
                'float',
                name='keep_prob',
            )

        self.use_precomputed_means = tf.placeholder(
                'bool',
                name='use_precomputed_means',
            )

        self.conv1 = self.conv_layer(
            input_=self.input_var,
            in_channels=1,
            out_channels=32,
            filter_size=5,
            stride=1,
            name="conv1",
        )

        self.pool1 = self.pool_layer(
            input_=self.conv1,
            ksize=2,
            stride=2,
            name="pool1",
        )

        self.conv2 = self.conv_layer(
            input_=self.pool1,
            in_channels=32,
            out_channels=64,
            filter_size=5,
            stride=1,
            name="conv2",
        )

        self.pool2 = self.pool_layer(
            input_=self.conv2,
            ksize=2,
            stride=2,
            name="pool2",
        )

        self.hidden = self.fc_layer(
            input_=self.pool2,
            in_dimension=7 * 7 * 64,
            out_dimension=1024,
            name="hidden",
        )

        self.dropout = self.dropout(self.hidden, self.keep_prob)

        self.readout = self.fc_layer(
            input_=self.dropout,
            in_dimension=1024,
            out_dimension=10,
            nonlinearity=tf.nn.softmax,
            name="readout",
        )

        self.loss = -tf.reduce_sum(self.corr_labels * tf.log(1e-13 + self.readout))
        optimizer = tf.train.AdamOptimizer(1e-3)
        grad_vars = optimizer.compute_gradients(self.loss)

        clipped_grad_vars = [(tf.clip_by_norm(grad, 10) if grad else None, var)
                for (grad, var) in grad_vars]

        for grad, var in clipped_grad_vars:
            tf.histogram_summary(var.name, var)
            if grad:
                tf.histogram_summary(var.name + "/clipgrad", grad)

        self.train = optimizer.apply_gradients(clipped_grad_vars)

        self.correct = tf.equal(tf.argmax(self.readout, 1), tf.argmax(self.corr_labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))

        tf.scalar_summary('loss', self.loss)
        tf.scalar_summary('accuracy', self.accuracy)
        tf.scalar_summary('log_error', tf.log(1 - self.accuracy))
