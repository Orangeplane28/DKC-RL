import gymnasium as gym
import time 
class TheRewards(gym.RewardWrapper):
    def __init__(self,env,living_bonus = 0.01, death_penalty = -1.00, timecheck = 15.0, fps = 30.0, idle_penalty = -0.10):
        super().__init__(env)
        self.living_bonus = living_bonus
        self.death_penalty = death_penalty
        self.prev_score = 0
        self.prev_lives = None
        self.idle_checker = int(timecheck * fps)
        self.idle_penalty = idle_penalty
        self.timeaftercheck = 0
        self.pointsaftercheck = 0

    def reset(self, **kwargs): #kwargs = keyword argument for seed and options
        obs, info = self.env.reset(**kwargs)
        self.prev_score = info.get("score", 0)
        self.prev_lives = info.get("lives", None)
        self.timeaftercheck = 0
        self.pointsaftercheck = self.prev_score
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated

        total = 0

        score = info.get("score", 0)
        score_delta = ((score - self.prev_score) % 256)/10 #score resets, goes back to 0 after 255
        total += score_delta

        if not done:
            total += self.living_bonus

        lives = info.get("lives", None)
        if (self.prev_lives is not None and lives is not None and lives < self.prev_lives):
            total += self.death_penalty

        self.timeaftercheck +=1
        if (self.timeaftercheck > self.idle_checker):
            if (score == self.pointsaftercheck):
                total += self.idle_penalty
                self.timeaftercheck = 0
                self.pointsaftercheck = score

        self.prev_score = info.get("score",0)
        self.prev_lives = lives

        return obs, reward + total, terminated, truncated, info
