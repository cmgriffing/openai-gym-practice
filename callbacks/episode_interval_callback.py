from rl.callbacks import Callback

class EpisodeIntervalCallback(Callback):

    def __init__(self, total_steps=0):
        self.total_steps = total_steps

    def on_episode_end(self, episode, logs={}):
        # print(logs)
        percent_complete = round(logs['nb_steps'] / self.total_steps * 100., 1)
        print('{}% Episode finished: {} reward: {} steps: {}'.format(percent_complete, episode, round(logs['episode_reward']), logs['nb_episode_steps']))
