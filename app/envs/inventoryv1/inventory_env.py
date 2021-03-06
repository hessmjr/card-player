import gym
from gym import spaces
from gym import utils
from gym.utils import seeding
import numpy as np


# noinspection PyAbstractClass
class InventoryEnv(gym.Env, utils.EzPickle):
    """Inventory control with lost sales environment
    This environment corresponds to the version of the inventoryv1 control
    with lost sales problem described in Example 1.1 in Algorithms for
    Reinforcement Learning by Csaba Szepesvari (2010).
    https://sites.ualberta.ca/~szepesva/RLBook.html
    """

    # noinspection PyMissingConstructor
    def __init__(self, n=100, k=5, c=2, h=2, p=3, lam=8):
        self.n = n
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Discrete(n)
        self.max = n
        self.state = n
        self.k = k
        self.c = c
        self.h = h
        self.p = p
        self.lam = lam

        # Set seed
        self.seed()

        # Start the first round
        self.reset()

    def demand(self):
        return np.random.poisson(self.lam)

    def transition(self, x, a, d):
        m = self.max
        return max(min(x + a, m) - d, 0)

    def reward(self, x, a, y):
        k = self.k
        m = self.max
        c = self.c
        h = self.h
        p = self.p
        r = -k * (a > 0) - c * max(min(x + a, m) - x, 0) - h * x + p * max(min(x + a, m) - y, 0)
        return r

    # noinspection PyAttributeOutsideInit
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        obs = self.state
        demand = self.demand()
        obs2 = self.transition(obs, action, demand)
        self.state = obs2
        # noinspection PyTypeChecker
        reward = self.reward(obs, action, obs2)
        done = 0
        return obs2, reward, done, {}

    def reset(self):
        return self.state
