import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Input
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import EpisodeParameterMemory, SequentialMemory

from callbacks.episode_interval_callback import EpisodeIntervalCallback
from callbacks.episode_batch_callback import EpisodeBatchCallback

from envs.lunar_lander_v2 import LunarLander

import argparse

parser = argparse.ArgumentParser(description='Run CEM Cartpole')
parser.add_argument('mode', metavar='mode', nargs=1,
                    help='What mode to start in? test or train')
parser.add_argument('label', metavar='label', nargs=1,
                    help='An extra string to label saved weights')
parser.add_argument('batch', metavar='batch', nargs=1,
                    help='The batch to load weights for during test mode.')

args = parser.parse_args()

print(args)

mode = args.mode[0]
if mode != 'train' and mode != 'test' :
  mode = 'train'

label = args.label[0] or ''
if label != '':
  label = f"_{label}"

test_batch = int(args.batch[0]) or 0

ENV_NAME = 'LunarLander-v2'

env = gym.make(ENV_NAME)
# env = LunarLander()



np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Option 1 : Simple model
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Activation('softmax'))

# Option 2: deep network
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2, input_shape=(1,) + env.observation_space.shape))
model.add(Dense(4, activation='linear'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# memory = EpisodeParameterMemory(limit=1000000, window_length=1)
memory = SequentialMemory(limit=1000000, window_length=1)

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)

# policy = BoltzmannQPolicy()

# cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
# cem.compile()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=2000, policy=policy, enable_double_dqn=True)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

agent = dqn

if mode == 'train':
  total_steps = 800000

  for batch in range(0, int(total_steps / 100000)):
    agent.fit(env, nb_steps=total_steps, visualize=True, verbose=0, callbacks=[EpisodeBatchCallback(total_steps=total_steps, current_batch=batch)], nb_max_episode_steps=1000)
    agent.save_weights('weights/{}{}_{}_params.h5f'.format(ENV_NAME, label, batch), overwrite=True)

if mode == 'test':

  agent.load_weights('weights/{}{}_{}_params.h5f'.format(ENV_NAME, label, test_batch))

  agent.test(env, nb_episodes=20, visualize=True)
