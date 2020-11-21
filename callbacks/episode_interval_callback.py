from rl.callbacks import Callback

class EpisodeIntervalCallback(Callback):

    def on_episode_end(self, episode, logs={}):
        print('Episode finished: {} reward: {}'.format(episode, logs['episode_reward']))
