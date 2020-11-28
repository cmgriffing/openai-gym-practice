from rl.callbacks import Callback

class VisualizerIntervalCallback(Callback):

    episode = 0
    episode_interval = 5

    def __init__(self, interval=5):
        self.episode_interval = interval

    def on_episode_end(self, episode, logs={}):
        self.episode += 1

    def on_action_end(self, action, logs):
        """ Render environment at the end of each action """
        if(self.episode % self.episode_interval == 0):
          self.env.render(mode='human')
