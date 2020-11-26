import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import EpisodeParameterMemory, SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
from rl.random import OrnsteinUhlenbeckProcess

from callbacks.episode_interval_callback import EpisodeIntervalCallback
from callbacks.episode_batch_callback import EpisodeBatchCallback
from callbacks.visualizer_interval_callback import VisualizerIntervalCallback

from envs.lunar_lander_original import LunarLander
from envs.car_racing import CarRacing

import argparse

parser = argparse.ArgumentParser(description='Run DDPG Car Racing')
parser.add_argument('mode', metavar='mode', nargs=1,
                    help='What mode to start in? test or train')
parser.add_argument('label', metavar='label', nargs=1,
                    help='An extra string to label saved weights')
parser.add_argument('batch', metavar='batch', nargs=1,
                    help='The batch to load weights for.')

args = parser.parse_args()

print(args)

mode = args.mode[0]
if mode != 'train' and mode != 'test' :
  mode = 'train'

label = args.label[0] or ''
if label != '':
  label = f"_{label}"

test_batch = int(args.batch[0]) or 0

ENV_NAME = 'CarRacing-v0'

# env = gym.make(ENV_NAME)
env = CarRacing()



np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

print(f'Number of Actions: {nb_actions}')

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16, activation='tanh'))
actor.add(Dense(16, activation='tanh'))
actor.add(Dense(16, activation='tanh'))
# actor.add(Dropout(0.2, input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(nb_actions, activation='tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32, activation='tanh')(x)
x = Dense(32, activation='tanh')(x)
x = Dense(32, activation='tanh')(x)
x = Dense(1, activation='tanh')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# memory = EpisodeParameterMemory(limit=1000000, window_length=1)
memory = SequentialMemory(limit=1000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

if mode == 'train':
  # total_steps = 800000
  total_steps = 200

  if test_batch > 0:
    agent.load_weights('weights/{}{}_batch_{}_params.h5f'.format(ENV_NAME, label, test_batch))

  agent.fit(
    env, nb_steps=total_steps, visualize=True, verbose=1, callbacks=[
      EpisodeBatchCallback(
        total_steps=total_steps, current_batch=test_batch
      ),
      # VisualizerIntervalCallback()
      # ModelIntervalCheckpoint('weights/{}{}_{}_params.h5f'.format(ENV_NAME, label, 0), 100000)
    ],
    nb_max_episode_steps=400
  )
  agent.save_weights('weights/{}{}_batch_{}_params.h5f'.format(ENV_NAME, label, test_batch + 1), overwrite=True)

if mode == 'test':

  agent.load_weights('weights/{}{}_batch_{}_params.h5f'.format(ENV_NAME, label, test_batch))

  agent.test(env, nb_episodes=20, visualize=True)
