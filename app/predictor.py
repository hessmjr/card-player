
class Predictor:
    def __init__(self):
        self.alpha = .2
        self.gamma = .4
        self.epsilon = .2
        self.epsilon_decay = 1.0

    def reset_state(self, observation, random_guess):
        pass

    def guess(self, prev_guess, observation, reward):

        pass

    def terminate(self):
        pass

class Learn:
    pass

class Observations:
    pass
