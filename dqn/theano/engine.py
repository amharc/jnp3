from model import Model
from game import Game
from replay import ReplayDB
from settings import settings
import numpy as np

import lasagne

model = Model(settings)
replay = ReplayDB(settings)

counter = 0

def epsilon(test=False):
    e0 = settings['initial_epsilon']
    e1 = settings['final_epsilon']
    lim = settings['epsilon_anneal_over']

    if test:
        return e1

    return e1 + max(0, (e0 - e1) * (lim - counter) / lim)

def train():
    minibatch = replay.sample()
    return model.train(
        observation=[x.old_state for x in minibatch],
        next_observation=[x.new_state for x in minibatch],
        rewards=[[x.reward] for x in minibatch],
        terminals=[[x.terminal] for x in minibatch],
        actions=[[x.action] for x in minibatch]
    )

while True:

    game = Game()
    observations = np.dstack((game.observe(),) * settings['phi_length'])[0]
    total_reward = 0

    for step in xrange(settings['max_steps']):
        counter += 1

        action = model.act(observations, epsilon())
        reward = game.act(action)
        observation = game.observe()
        terminal = step == settings['max_steps'] - 1

        replay.push(
                observation=observation,
                reward=reward,
                action=action,
                terminal=terminal,
            )

        if len(replay) >= settings['replay_start']:
            loss = train()

        observations = np.hstack((np.expand_dims(observation, 2), observations[:,1:]))
        total_reward += reward

        if counter % 100 == 0:
            print("counter={}, reward={}, epsilon={}, replay={}, loss={}".format(counter, reward, epsilon(), len(replay), loss))
            np.savez('model.npz', *lasagne.layers.get_all_param_values(model.q_network))
