import sys
sys.path.append("ale")

import numpy as np
import tensorflow as tf
import random
import gc

from model import Model
from replay import ReplayDB
from emulator import Emulator
from settings import settings

class Engine(object):
    def __init__(self):
        self.session = tf.InteractiveSession()

        self.emulator = Emulator(settings)
        settings['num_actions'] = len(self.emulator.actions)
        self.replay = ReplayDB(settings)

        with tf.variable_scope('model'):
            self.model = Model(settings)

        self.summary = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter('summary-log', self.session.graph_def)

        self.session.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver(max_to_keep=1000000)
        checkpoint = tf.train.get_checkpoint_state("networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoint: {}".format(checkpoint.model_checkpoint_path))
        else:
            print("Unable to load checkpoint")

        self.summary_cnt = 0
        self.episode_cnt = 0
        self.timer = self.session.run(self.model.global_step)
        self.no_op = tf.no_op()

    def epsilon(self, test=False):
        e0 = settings['initial_epsilon']
        e1 = settings['final_epsilon']
        lim = settings['epsilon_anneal_length']

        if test:
            return e1

        return e1 + max(0, (e0 - e1) * (lim - self.timer) / lim)

    def choose_action(self, test=False):
        if np.random.rand() < self.epsilon(test):
            return random.randrange(len(self.emulator.actions)) 
        else:
            predictions = self.model.act_network.readout.eval({
                self.model.images: [self.images]
            })[0]
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
            terminal = self.emulator.terminal()

            if not test:
                self.replay.push(
                        image=image,
                        reward=reward,
                        action=action,
                        terminal=terminal
                    )

            if push_to is not None:
                push_to.append(action)

            if terminal:
                break

            if not test and len(self.replay) >= settings['replay_start']:
                if updates % settings['update_frequency'] == 0:
                    self.train()
                updates += 1

            self.images = np.dstack((image, self.images[:,:,1:]))
            total_reward += reward

        if not test:
            self.episode_cnt += 1
            if len(self.replay) >= settings['replay_start']:
                self.writer.flush()

            if self.episode_cnt % settings['save_every_episodes'] == 0:
                self.saver.save(self.session, 'networks/checkpoint', global_step=self.timer)

        return total_reward

    def train(self):
        minibatch = self.replay.sample()
        action_mask = np.zeros((len(minibatch), settings['num_actions']))

        for i, sample in enumerate(minibatch):
            action_mask[i][sample.action] = 1

        with_summary = self.summary_cnt % settings['write_summary_every'] == 0
        self.summary_cnt += 1

        _, summary = self.session.run([
            self.model.train_op,
            self.summary if with_summary else self.no_op],
            {
                self.model.images: [x.old_state for x in minibatch],
                self.model.next_images: [x.new_state for x in minibatch],
                self.model.action_mask: action_mask,
                self.model.rewards: [x.reward for x in minibatch],
                self.model.terminals: [x.terminal for x in minibatch]
            }
        )

        if with_summary:
            self.writer.add_summary(summary, self.timer)

        self.timer += 1

        if self.timer % settings['target_update_freq'] == 0:
            self.session.run(self.model.reset_target_op)

if __name__ == '__main__':
    engine = Engine()
    for i in xrange(10000000):
        reward = engine.episode()
        print("Episode {}: epsilon = {}, reward = {}, replay = {}, timer = {}".format(i, engine.epsilon(), reward, len(engine.replay), engine.timer))

        if i % settings['test_every_episodes'] == 0:
            for _ in xrange(8):
                actions = []
                test_reward = engine.episode(test=True, push_to=actions)
                print("Tested: {}: {}".format(test_reward, actions))
            gc.collect()
