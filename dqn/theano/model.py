import numpy as np
import theano
import theano.tensor as T

import lasagne

def make_network(settings):
    last_layer = lasagne.layers.InputLayer(
                shape=(None, settings['input_length'], settings['phi_length']),
            )

    for num_units in settings['layers']:
        last_layer = lasagne.layers.DenseLayer(
                last_layer,
                num_units=num_units,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotNormal(gain='relu'),
                b=lasagne.init.Constant(0.1)
            )
    
    l_out = lasagne.layers.DenseLayer(
            last_layer,
            num_units=settings['num_actions'],
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(0.1)
        )

    return l_out


class Model(object):
    def __init__(self, settings):
        self.settings = settings

        observation = T.tensor3('observation')
        next_observation = T.tensor3('next_observation')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')

        self.observation = theano.shared(np.zeros(
            (settings['batch_size'],
             settings['phi_length'],
             settings['input_length'],
             ),
             dtype=theano.config.floatX))

        self.next_observation = theano.shared(np.zeros(
            (settings['batch_size'],
             settings['phi_length'],
             settings['input_length'],
             ),
             dtype=theano.config.floatX))

        self.rewards = theano.shared(np.zeros(
            (settings['batch_size'], 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions = theano.shared(np.zeros(
            (settings['batch_size'], 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals = theano.shared(np.zeros(
            (settings['batch_size'], 1), dtype='int32'),
            broadcastable=(False, True))

        self.q_network = make_network(settings)
        self.target_network = make_network(settings)

        scores = lasagne.layers.get_output(self.q_network, observation)
        target_scores = theano.gradient.disconnected_grad(
                lasagne.layers.get_output(self.target_network, next_observation)
            )

        future_rewards = settings['discount'] * T.max(target_scores, axis=1, keepdims=True)
        future_rewards = future_rewards * (T.ones_like(terminals) - terminals)
        target = rewards + future_rewards
        diff = target - scores[T.arange(settings['batch_size']), actions.reshape((-1,))].reshape((-1, 1))
        loss = T.mean(diff ** 2)

        loss += lasagne.regularization.regularize_network_params(self.q_network, lasagne.regularization.l2) * settings['l2_regularisation']

        params = lasagne.layers.helper.get_all_params(self.q_network)
        #updates = lasagne.updates.rmsprop(loss, params, settings['rms_learning_rate'],
        #        settings['rms_decay'], settings['rms_epsilon'])

        updates = lasagne.updates.sgd(loss, params, settings['sgd_learning_rate'])

        self._train = theano.function([], [loss, scores], updates=updates,
                givens = {observation: self.observation,
                          next_observation: self.next_observation,
                          rewards: self.rewards,
                          actions: self.actions,
                          terminals: self.terminals
                         })

        self._predict = theano.function([], scores, givens={observation: self.observation})

        self.counter = 0

    def update_target_network(self):
        lasagne.layers.helper.set_all_param_values(
            self.target_network,
            lasagne.layers.helper.get_all_param_values(
                self.q_network
            )
        )

    def predict(self, observation):
        observations = np.zeros(
                (self.settings['batch_size'], self.settings['input_length'], self.settings['phi_length']),
                dtype=theano.config.floatX)
        observations[0, ...] = observation
        self.observation.set_value(observations)
        return self._predict()[0]

    def act(self, observation, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.settings['num_actions'])

        predictions = self.predict(observation)
        return np.argmax(predictions)

    def train(self, observation, next_observation, actions, rewards, terminals):
        self.observation.set_value(observation)
        self.next_observation.set_value(next_observation)
        self.actions.set_value(actions)
        self.rewards.set_value(rewards)
        self.terminals.set_value(terminals)

        if self.counter % self.settings['target_update_frequency'] == 0:
            self.update_target_network()

        self.counter += 1

        loss, _ = self._train()
        return np.sqrt(loss)
