import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import EpisodeParameterMemory, SequentialMemory

import argparse

parser = argparse.ArgumentParser(description='Run CEM Cartpole')
parser.add_argument('mode', metavar='mode', nargs=1,
                    help='What mode to start in? test or train')
args = parser.parse_args()

print(args)

mode = args.mode[0]
if mode != 'train' and mode != 'test' :
  mode = 'train'

ENV_NAME = 'Acrobot-v1'

env = gym.make(ENV_NAME)


np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Option 1 : Simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

# Option 2: deep network
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('softmax'))

print(model.summary())

memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()

agent = cem

if mode == 'train':

  agent.fit(env, nb_steps=100000, visualize=True, verbose=1)

  agent.save_weights('weights/cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)

if mode == 'test':

  agent.load_weights('weights/cem_{}_params.h5f'.format(ENV_NAME))

  agent.test(env, nb_episodes=5, visualize=True)
