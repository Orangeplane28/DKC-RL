import numpy as np, gymnasium as gym
from gymnasium import spaces

class ActionDWrapper(gym.ActionWrapper):
    def __init__(self,env):
        super().__init__(env) #this wrapper will inherit from superclass ActionWrapper
        self.buttons = getattr(env, "buttons", ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R'])
        self.actions = [[], ['RIGHT'], ['LEFT'], ['B'], ['RIGHT', 'B'], ['Y'], ['RIGHT', 'Y']]
        self.action_space = spaces.Discrete(len(self.actions)) #all possible ai action outcomes

    def cnvrtAction(self, nAction):
        out = np.zeros(len(self.buttons), dtype=np.int8) #controller vector of 12 possible inputs
        for i in self.actions[nAction]:
            out[self.buttons.index(i)] = 1 #what combination of array is chosen
        return out

    def action(self, act: int):
        return self.cnvrtAction(act)
