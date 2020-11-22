from rl.callbacks import Callback

class EpisodeBatchCallback(Callback):

    def __init__(self, total_steps=0, current_batch=0):
        self.total_steps = total_steps
        self.current_batch = current_batch

    def on_episode_end(self, episode, logs={}):
        # print(logs)
        current_steps = logs['nb_steps'] + (100000 * self.current_batch)
        percent_complete = round(current_steps / self.total_steps * 100., 1)
        print('{}% Episode finished: {} reward: {} steps: {}'.format(percent_complete, episode, round(logs['episode_reward']), logs['nb_episode_steps']))
