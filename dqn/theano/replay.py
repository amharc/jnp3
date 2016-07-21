import numpy as np
from collections import namedtuple
import theano

Replay = namedtuple('Replay', 'old_state new_state action reward terminal')

class TerminalException(Exception):
    pass

class ReplayDB(object):
    def __init__(self, settings):
        self.settings = settings
        self.length = settings['replay_length']

        self.observations = np.full((self.length, settings['input_length']), 42.0, dtype=theano.config.floatX)
        self.rewards = np.empty((self.length,), dtype=theano.config.floatX)
        self.actions = np.empty((self.length,), dtype=np.int8)
        self.terminals = np.empty((self.length,), dtype=np.int8)

        self.end = 0
        self.wrapped = False

    def push(self, observation, action, reward, terminal):
        self.observations[self.end] = observation
        self.actions[self.end] = action
        self.rewards[self.end] = reward
        self.terminals[self.end] = terminal

        self.end += 1
        if self.end == self.length:
            self.end = 0
            self.wrapped = True

    def take_from(self, pos):
        endpos = pos + self.settings['phi_length']

        prev_terminals = np.take(
                self.terminals,
                xrange(pos, endpos),
                axis=0,
                mode='wrap'
            )

        if np.any(prev_terminals):
            raise TerminalException()

        old_state = np.take(
                self.observations,
                indices=xrange(pos, endpos),
                axis=0,
                mode='wrap',
            )

        new_state = np.take(
                self.observations,
                xrange(pos + 1, endpos + 1),
                axis=0,
                mode='wrap',
            )

        reward = np.take(self.rewards, endpos, axis=0, mode='wrap')
        action = np.take(self.actions, endpos, axis=0, mode='wrap')
        terminal = np.take(self.terminals, endpos, axis=0, mode='wrap')

        return Replay(old_state=np.dstack(old_state)[0], new_state=np.dstack(new_state)[0],
                      reward=reward, action=action, terminal=terminal)

    def __len__(self):
        if self.wrapped:
            return self.length
        else:
            return self.end

    def sample(self):
        results = []

        while True:
            limit = self.length if self.wrapped else self.end - self.settings['phi_length']
            sampled = np.random.randint(0, limit, 2 * self.settings['batch_size'])

            for sample in sampled:
                while True:
                    if len(results) == self.settings['batch_size']:
                        return results

                    try:
                        results.append(self.take_from(sample))
                    except TerminalException:
                        break
