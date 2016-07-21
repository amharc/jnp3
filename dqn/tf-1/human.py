from emulator import Emulator
from collections import deque, namedtuple
import random
import itertools
import numpy as np
import cv2
import time

if __name__ == '__main__':
    emulator = Emulator(rom='SPCINVAD.BIN')

    cv2.startWindowThread()
    cv2.namedWindow("preview")

    emulator.reset()

    reward_episode = 0

    print("Num frames per episode: {}".format(emulator.max_num_frames_per_episode))

    for frame in range(emulator.max_num_frames_per_episode):
        action_idx = int(input())
        reward = emulator.act(emulator.actions[action_idx])
        print("Instead: {}, i.e. {}, reward = {}".format(action_idx, emulator.actions[action_idx], reward))
        
        if emulator.terminal():
            break

        reward_episode += reward

        actions = np.zeros([len(emulator.actions)])
        actions[action_idx] = 1

        new_images = np.dstack((np.reshape(emulator.image(), (80, 80, 1)), images[:,:,1:]))

        images = new_images

        cv2.imshow('preview', emulator.image())
