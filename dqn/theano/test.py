from model import Model
from game import Game
from settings import settings
import numpy as np
import cv2

import lasagne

model = Model(settings)

with np.load('model.npz') as f:
    params = [f['arr_%d' % i] for i in xrange(len(f.files))]
lasagne.layers.set_all_param_values(model.q_network, params)
model.update_target_network()

cv2.startWindowThread()
cv2.namedWindow("preview")

while True:
    game = Game()
    observations = np.dstack((game.observe(),) * settings['phi_length'])[0]
    for step in xrange(settings['max_steps']):
        predictions = model.predict(observations)
        print predictions
        action = model.act(observations, settings['test_epsilon'])
        reward = game.act(action)
        observation = game.observe()
        terminal = step == settings['max_steps'] - 1

        game.draw()
        observations = np.hstack((np.expand_dims(observation, 2), observations[:,1:]))
