import gymnasium as gym
import numpy as np
import cv2

class PreprocessFrame(gym.ObservationWrapper):

    def __init__(self, env, width=84, height=84):
        super().__init__(env) #inherit superclass
        self.w, self.h = width, height
        # Update the space with decided format and color
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.h, self.w, 1), dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) #raw RGB frame to black and white
        resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        #make image smaller, interarea= down scale
        return resized[..., None]   # adds new axis = H, W, 1 = correct format for CNN
