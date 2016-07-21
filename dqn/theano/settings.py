import game

settings = {
    'phi_length': 2,
    'input_length': game.OBSERVATION_LENGTH,
    'num_actions': len(game.ACTIONS),
    'batch_size': 64,
    'rms_learning_rate': 0.001,
    'rms_decay': 0.95,
    'rms_epsilon': 0.001,
    'sgd_learning_rate': 0.001,
    'discount': 0.70,
    'target_update_frequency': 100,
    'replay_length': 1000000,
    'initial_epsilon': 1,
    'final_epsilon': 0.05,
    'test_epsilon': 0.05,
    'epsilon_anneal_over': 100000,
    'replay_start': 100,
    'max_steps': 10000,
    'l2_regularisation': 0.01,
    'layers': [50, 50],
}
