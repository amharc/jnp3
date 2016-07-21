from ale_python_interface import ALEInterface
import cv2
import numpy as np

class Emulator(object):
    FRAME_SKIP = 4
    SCREEN_WIDTH = 84
    SCREEN_HEIGHT = 84

    def __init__(self, rom):
        self.ale = ALEInterface()
        self.max_num_frames_per_episode = 100000 #self.ale.getInt('max_num_frames_per_episode')
        self.ale.setInt('frame_skip', self.FRAME_SKIP)
        self.ale.loadROM('roms/' + rom)
        self.actions = self.ale.getMinimalActionSet()
        
    def reset(self):
        self.ale.reset_game()

    def image(self):
        screen = self.ale.getScreenGrayscale()
        screen = cv2.resize(screen, (self.SCREEN_HEIGHT, self.SCREEN_WIDTH))
        return np.reshape(screen, (self.SCREEN_HEIGHT, self.SCREEN_WIDTH))

    def act(self, action):
        return self.ale.act(action)

    def terminal(self):
        return self.ale.game_over()
