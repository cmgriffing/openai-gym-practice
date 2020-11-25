from rl.callbacks import Callback

class VisualizerIntervalCallback(Callback):

    episode = 0

    def on_episode_end(self, episode, logs={}):
        self.episode += 1

    def on_action_end(self, action, logs):
        """ Render environment at the end of each action """
        if(self.episode % 10 == 0):
          self.env.render(mode='human')
