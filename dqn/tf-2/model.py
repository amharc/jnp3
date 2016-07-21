import tensorflow as tf
import numpy as np
from collections import namedtuple

Network = namedtuple("Network", "conv1 conv2 conv3 hidden readout variables")

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
                padding="VALID",
                **kwargs
            )

    @staticmethod
    def max_pool(input_, ksize, stride, **kwargs):
        return tf.nn.max_pool(
                input_,
                ksize=[1, ksize, ksize, 1],
                strides=[1, stride, stride, 1],
                padding="VALID",
                **kwargs
            )


    def conv_layer(self, input_, in_channels, out_channels, filter_size, stride, name, variables,
                   pool_ksize=2, pool_stride=2):
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

            variables.append(weights)
            variables.append(bias)

            conv = self.conv(input_, weights, stride, name="conv")
            biased = conv + bias
            relu = tf.nn.relu(biased)

            tf.histogram_summary(name + "/conv", conv)
            tf.histogram_summary(name + "/biased", biased)
            tf.histogram_summary(name + "/relu", relu)

            return relu

    def fc_layer(self, input_, in_dimension, out_dimension, name, variables):
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

            variables.append(weights)
            variables.append(bias)

            reshaped = tf.reshape(input_, [-1, in_dimension])
            matmul = tf.matmul(reshaped, weights) 
            biased = matmul + bias
            relu = tf.nn.relu(biased)

            tf.histogram_summary(name + "/matmul", matmul)
            tf.histogram_summary(name + "/biased", biased)
            tf.histogram_summary(name + "/relu", relu)

            return relu

    def readout_layer(self, input_, in_dimension, out_dimension, name, variables):
        with tf.variable_scope(name):
            weights = self.weight_var(
                    [in_dimension, out_dimension],
                    gain=1.0,
                    name="weights"
                )

            bias = self.bias_var(
                    [out_dimension],
                    name="bias"
                )

            variables.append(weights)
            variables.append(bias)

            reshaped = tf.reshape(input_, [-1, in_dimension])
            matmul = tf.matmul(reshaped, weights) 
            biased = matmul + bias

            tf.histogram_summary(name + "/matmul", matmul)
            tf.histogram_summary(name + "/biased", biased)

            return biased

    def _init_network(self, input_, name):
        variables = []

        conv1 = self.conv_layer(
            input_=input_,
            in_channels=self.settings['phi_length'],
            name=name + "/conv1",
            variables=variables,
            **self.settings['conv1']
        )

        conv2 = self.conv_layer(
            input_=conv1,
            in_channels=self.settings['conv1']['out_channels'],
            name=name + "/conv2",
            variables=variables,
            **self.settings['conv2']
        )

        conv3 = self.conv_layer(
            input_=conv2,
            in_channels=self.settings['conv2']['out_channels'],
            name=name + "/conv3",
            variables=variables,
            **self.settings['conv3']
        )

        hidden = self.fc_layer(
            input_=conv3,
            name=name + "/hidden",
            variables=variables,
            **self.settings['hidden']
        )

        readout = self.readout_layer(
            input_=hidden,
            name=name + "/readout",
            in_dimension=self.settings['hidden']['out_dimension'],
            out_dimension=self.settings['num_actions'],
            variables=variables,
        )

        return Network(
            conv1=conv1,
            conv2=conv2,
            conv3=conv3,
            hidden=hidden,
            readout=readout,
            variables=variables
        )

    def image_var(self, **kwargs):
        return tf.placeholder(
                'float',
                [ None
                , self.settings['screen_height']
                , self.settings['screen_width']
                , self.settings['phi_length']
                ],
                **kwargs
            )

    def _init_vars(self):
        self.images = self.image_var(name="images")
        self.next_images = self.image_var(name="next_images")

        self.action_mask = tf.placeholder("float", [None, self.settings['num_actions']], name="action_mask")
        self.rewards = tf.placeholder("float", [None], name="rewards")
        self.terminals = tf.placeholder("float", [None], name="terminals")

    def _init_train(self):
        readout = tf.stop_gradient(self.target_network.readout)

        # 0 if terminal, max(prediction) if not
        future_rewards = tf.reduce_max(readout, reduction_indices=[1,]) * (1 - self.terminals)
        tf.histogram_summary("rewards_future", future_rewards)

        wanted = self.rewards + self.settings['discount'] * future_rewards
        tf.histogram_summary("rewards_wanted", wanted)

        current = tf.reduce_sum(
                self.act_network.readout * self.action_mask,
                reduction_indices=[1,],
                name="rewards_current"
            )
        tf.histogram_summary("rewards_current", current)

        loss = tf.square(current - wanted)
        self.error = tf.reduce_sum(loss, name="prediction_error")

        tf.scalar_summary('error', self.error)

        grad_vars = self.settings['optimizer'].compute_gradients(self.error)

        clipped_grad_vars = [(tf.clip_by_norm(grad, 10) if grad else None, var)
                for (grad, var) in grad_vars]

        for grad, var in clipped_grad_vars:
            tf.histogram_summary(var.name, var)
            if grad:
                tf.histogram_summary(var.name + "_clipgrad", grad)

        self.train_op = self.settings['optimizer'].apply_gradients(clipped_grad_vars, global_step=self.global_step)


    def __init__(self, settings):
        self.settings = settings

        self.global_step = tf.Variable(1, name='global_step', trainable=False)

        with tf.variable_scope("vars"):
            self._init_vars()

        with tf.variable_scope("network/active"):
            self.act_network = self._init_network(self.images, "act")

        with tf.variable_scope("network/target"):
            self.target_network = self._init_network(self.next_images, "target")

        with tf.variable_scope("train"):
            self._init_train()

        self.reset_target_op = tf.group(*(
            target_var.assign(act_var)
            for act_var, target_var in zip(self.act_network.variables, self.target_network.variables)
        ))
