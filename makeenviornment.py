import retro
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation as FrameStack
from gymnasium.wrappers import TimeLimit
from ActionDWrapper import ActionDWrapper
from Preprocess import PreprocessFrame
from TheRewards import TheRewards

def make_dkc_env(
    state="1Player.CongoJungle.JungleHijinks.Level1.state",
    scenario="scenario.json",
    width=84,
    height=84,
    frame_stack=4,
    max_episode_steps=11000,  
    render_mode= None):

    env = retro.make(
        game="DonkeyKongCountry-Snes",
        #state=state,
        #scenario=scenario,
        render_mode=render_mode
    )
    env = ActionDWrapper(env)
    env = PreprocessFrame(env, width, height)
    env = FrameStack(env, frame_stack) #AI sees velocity
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = TheRewards(env, living_bonus=0.01)
    return env

if __name__ == "__main__": #only ran from this file directly, not if imported
    env = make_dkc_env()
    obs, info = env.reset()

    try:
        for t in range(11000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            #env.render()

            if (t % 500) == 0:
                print(obs, reward, terminated, info)

            if terminated or truncated:
                print("Episode ended")
                obs, info = env.reset()

    finally:
        env.close()
        print("Closed env.")
