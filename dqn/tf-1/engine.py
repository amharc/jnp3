from emulator import Emulator
from network import Network
from collections import deque, namedtuple
import random
import itertools
import tensorflow as tf
import numpy as np

REPLAY_LEN = 1000000
PHI_FRAMES = 4

INITIAL_EPS = 1
FINAL_EPS = 0.05

DECAY = 0.95

OBSERVE = 50
RECORD = 0
EXPLORE = 1000000

BATCH_SIZE = 32

EPISODES = 200000

SUMMARY_DELTA = 100

#LEARNING_RATE = 1e-6

Replay = namedtuple('Replay', 'old_state new_state action reward terminal')

def choose_action(images, epsilon):
    if np.random.rand() < epsilon:
       return random.randrange(len(emulator.actions)) 
    else:
       predictions = network.readout.eval(feed_dict = {
           network.image: [images]
        })[0]
       action_idx = np.argmax(predictions)
       selected_actions.append(action_idx)
       return action_idx

def anneal(timer):
    return FINAL_EPS + max(0, (INITIAL_EPS - FINAL_EPS) * (EXPLORE - timer) / EXPLORE)

summary_timer = 0

def train():
    minibatch = random.sample(replay, BATCH_SIZE)

    predictions = network.readout.eval(feed_dict = {
        network.image: [x.new_state for x in minibatch]
    })

    rewards = [
            [x.reward + DECAY * np.max(pred) if not x.terminal else x.reward]
            for x, pred in zip(minibatch, predictions)
    ]

    def mk_actions(action_idx):
        actions = np.zeros([len(emulator.actions)])
        actions[action_idx] = 1
        return actions

    feed_dict = {
        network.image: [x.old_state for x in minibatch],
        network.actions: [mk_actions(x.action) for x in minibatch],
        network.rewards: rewards
    }

    global summary_timer
    summary_timer += 1
    if timer > RECORD and summary_timer % SUMMARY_DELTA == 0:
        writer.add_summary(
            session.run(merged, feed_dict=feed_dict),
            timer
        )

    train_op.run(feed_dict=feed_dict)

if __name__ == '__main__':
    replay = deque([], maxlen=REPLAY_LEN)
    emulator = Emulator(rom='breakout.bin')

    session = tf.InteractiveSession()
    with tf.variable_scope("network"):
        network = Network(len(emulator.actions))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    #optimizer  = tf.train.RMSPropOptimizer(0.0002, 0.99, 0.99, 1e-6)
    optimizer  = tf.train.AdamOptimizer(1e-5)
    grads_vars = optimizer.compute_gradients(network.cost)
    clipped    = [(tf.clip_by_norm(g, 5, name="clip_grads"), v) for (g, v) in grads_vars]
    for g, v in clipped:
        tf.histogram_summary("clipped_grad_" + v.name, g)
    train_op   = optimizer.apply_gradients(clipped, global_step=global_step, name="apply_grads")
    #optimizer = tf.train.AdamOptimizer(25e-5, epsilon=0.1).minimize(network.cost)

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('summary-log', session.graph_def)

    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
        print("Loaded checkpoint: {}".format(checkpoint.model_checkpoint_path))
    else:
        print("Unable to load checkpoint")

    timer = session.run(global_step)
    for episode in range(EPISODES):
        emulator.reset()
        images = emulator.image()
        images = np.dstack((images,) * PHI_FRAMES)

        print("----------------- EPISODE {} -----------------".format(episode))

        reward_episode = 0

        selected_actions = deque()

        for frame in range(emulator.max_num_frames_per_episode):
            epsilon = anneal(timer)
            action_idx = choose_action(images, epsilon)

            reward = emulator.act(emulator.actions[action_idx])
            
            if emulator.terminal():
                break

            reward_episode += reward

            new_images = np.dstack((np.reshape(emulator.image(), (84, 84, 1)), images[:,:,1:]))

            replay.append(Replay(
                old_state = np.copy(images),
                new_state = np.copy(new_images),
                action = action_idx,
                reward = reward,
                terminal = emulator.terminal()
            ))

            if len(replay) > OBSERVE:
                train()

            images = new_images

            timer += 1

        saver.save(session, 'networks/checkpoint', global_step=timer)
        writer.flush()
        print("----------------- EPISODE {} : reward = {}, timer = {}, epsilon = {}, actions = {}".format(episode, reward_episode, timer, anneal(timer), selected_actions))
