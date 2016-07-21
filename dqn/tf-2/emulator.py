from ale_python_interface import ALEInterface
import cv2
import numpy as np

class Emulator(object):
    def __init__(self, settings):
        self.ale = ALEInterface()
        self.ale.setInt('frame_skip', settings['frame_skip'])
        self.ale.setInt('random_seed', np.random.RandomState().randint(1000))
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM('roms/' + settings['rom_name'])
        self.actions = self.ale.getMinimalActionSet()
        self.width = settings['screen_width']
        self.height = settings['screen_height']
        
    def reset(self):
        self.ale.reset_game()

    def image(self):
        screen = self.ale.getScreenGrayscale()
        screen = cv2.resize(screen, (self.height, self.width),
                interpolation=cv2.INTER_LINEAR)
        return np.reshape(screen, (self.height, self.width))

    def full_image(self):
        screen = self.ale.getScreenRGB()
        return screen

    def act(self, action):
        return self.ale.act(self.actions[action])

    def terminal(self):
        return self.ale.game_over()
