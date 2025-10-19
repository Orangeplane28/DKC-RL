import os
import retro
import gymnasium as gym
from gymnasium.wrappers import TransformObservation
from gymnasium.wrappers import TimeLimit
from ActionDWrapper import ActionDWrapper
from Preprocess import PreprocessFrame
from TheRewards import TheRewards
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

def make_env(
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
        render_mode=None
    )
    env = ActionDWrapper(env)
    env = PreprocessFrame(env, width, height)
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = TheRewards(env, living_bonus=0.01)
    return env

def env_fn(seed):
    def thunk():
        env = make_env()
        env.reset(seed=seed)
        return env
    return thunk

def main():
    os.makedirs("logs/tb", exist_ok=True)
    os.makedirs("ckpts", exist_ok=True) #will store saved models

    train_env = SubprocVecEnv([env_fn(i) for i in range(4)])  #create parallel game enviornments need to be as functions
    train_env = VecFrameStack(train_env, n_stack=4)    # (84,84,4)
    train_env = VecTransposeImage(train_env)  
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([env_fn(10000)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecMonitor(eval_env)

    model = PPO(
        "CnnPolicy",                 # built in PPO with CNN and Actor-Critic policy
        train_env,
        n_steps=256, batch_size=1024, n_epochs=4,
        gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        learning_rate=2.5e-4, max_grad_norm=0.5,
        tensorboard_log="logs/tb",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="ckpts",
        log_path="logs",
        eval_freq=20000,         
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=5000000, callback=eval_cb)
    model.save("ckpts/ppo_dk_final")

    train_env.close()
    eval_env.close()

if __name__ == "__main__": #only ran from this file directly, not if imported
    main()

