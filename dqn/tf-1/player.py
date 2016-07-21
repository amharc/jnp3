from emulator import Emulator
from network import Network
from collections import deque, namedtuple
import random
import itertools
import tensorflow as tf
import numpy as np
import cv2
import time

EPSILON = 0.05
PHI_FRAMES = 4

def choose_action(images, epsilon):
    if np.random.rand() < epsilon:
       return random.randrange(len(emulator.actions)) 
    else:
       readout = network.readout.eval(feed_dict = {
           network.image: [images]
        })[0]

       print(readout)
       print("NEURAL NETWORK SUGGESTS: {}".format(np.argmax(readout)))
       return np.argmax(readout)

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        emulator = Emulator(rom='breakout.bin')

        session = tf.InteractiveSession()
        with tf.variable_scope("network"):
            network = Network(len(emulator.actions))

        optimizer = tf.train.RMSPropOptimizer(0.0002, 0.99, 0.99, 1e-6).minimize(network.cost)
        #optimizer = tf.train.AdamOptimizer(1e-6).minimize(network.cost)

        session.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoint: {}".format(checkpoint.model_checkpoint_path))
        else:
            print("Unable to load checkpoint")

        cv2.startWindowThread()
        cv2.namedWindow("preview")

        emulator.reset()
        images = emulator.image()
        images = np.stack((images,) * PHI_FRAMES, axis = 2)

        reward_episode = 0


        print("Num frames per episode: {}".format(emulator.max_num_frames_per_episode))

        for frame in range(emulator.max_num_frames_per_episode):
            action_idx = choose_action(images, EPSILON)
            reward = emulator.act(emulator.actions[action_idx])

            print("Action: {}, i.e. {}, reward = {}".format(action_idx, emulator.actions[action_idx], reward))
            
            if emulator.terminal():
                break

            reward_episode += reward

            actions = np.zeros([len(emulator.actions)])
            actions[action_idx] = 1

            new_images = np.dstack((np.reshape(emulator.image(), (84, 84, 1)), images[:,:,1:]))

            images = new_images

            cv2.imshow('preview', emulator.image())
            time.sleep(0.2)
