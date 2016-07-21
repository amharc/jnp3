import tensorflow as tf

settings = {
    'screen_width': 84,
    'screen_height': 84,
    'phi_length': 4,
    'frame_skip': 4,
    'rom_name': 'breakout.bin',
    'conv1': {
        'out_channels': 32,
        'filter_size': 8,
        'stride': 4,
    },
    'conv2': {
        'out_channels': 64,
        'filter_size': 4,
        'stride': 2,
    },
    'conv3': {
        'out_channels': 64,
        'filter_size': 3,
        'stride': 1,
    },
    'hidden': {
        'in_dimension': 3136,
        'out_dimension': 256,
    },
    'discount': 0.99,
    'optimizer': tf.train.RMSPropOptimizer(0.00025, 0.95, 0.95, 0.01),
#    'optimizer': tf.train.AdamOptimizer(1e-6),
    'replay_length': 1000000,
    'batch_size': 32,
    'write_summary_every': 200,
    'target_update_freq': 1,
    'initial_epsilon': 1,
    'final_epsilon': 0.1,
    'replay_start': 50000,
    'epsilon_anneal_length': 1000000,
    'test_every_episodes': 50,
    'save_every_episodes': 50,
    'update_frequency': 4,
}
