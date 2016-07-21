import sys
sys.path.append("ale")

import numpy as np
import tensorflow as tf
import random
import gc
import cv2
import time

from model import Model
from replay import ReplayDB
from emulator import Emulator
from settings import settings

class Visualize(object):
    def __init__(self):
        self.session = tf.InteractiveSession()

        self.emulator = Emulator(settings)
        settings['num_actions'] = len(self.emulator.actions)

        with tf.variable_scope('model'):
            self.model = Model(settings)

        self.session.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver(max_to_keep=1000000)
        checkpoint = tf.train.get_checkpoint_state("networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoint: {}".format(checkpoint.model_checkpoint_path))
        else:
            raise RuntimeError("Unable to load checkpoint")

        cv2.startWindowThread()
        cv2.namedWindow("preview")
        cv2.namedWindow("full")

    def epsilon(self, test=False):
        return settings['final_epsilon']

    def choose_action(self, test=False):
        if np.random.rand() < self.epsilon(test):
            return random.randrange(len(self.emulator.actions)) 
        else:
            predictions = self.model.act_network.readout.eval({
                self.model.images: [self.images]
            })[0]
            print predictions, np.argmax(predictions)
            return np.argmax(predictions)

    def episode(self, test=False, push_to=None):
        self.emulator.reset()
        self.images = np.dstack((self.emulator.image(),) * settings['phi_length'])

        total_reward = 0
        updates = 0

        while True:
            action = self.choose_action(test)
            reward = self.emulator.act(action)
            image = self.emulator.image()
            cv2.imshow('preview', image)
            cv2.imshow('full', self.emulator.full_image())
            terminal = self.emulator.terminal()

            if reward > 0:
                print "reward:", reward

            if terminal:
                break

            self.images = np.dstack((image, self.images[:,:,1:]))
            total_reward += reward

            time.sleep(0.1)

        return total_reward

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        engine = Visualize()
        reward = engine.episode()
        print("Episode: epsilon = {}, reward = {}".format(engine.epsilon(), reward))
