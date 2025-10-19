import gymnasium as gym
import time
import retro
import numpy as np
GAME_ID = "DonkeyKongCountry-Snes"


env = retro.make(GAME_ID) #make creates training enviornment 

buttons = ['B','Y','Select','Start','Up','Down','Left','Right','A','X','L','R']
actions = [[], ['Right'], ['Left'], ['B'], ['Right', 'B'], ['Y'], ['Right', 'Y']]

def cnvrtAction(nAction):
    out = [0,0,0,0,0,0,0,0,0,0,0,0] #12 possible inputs
    for i in actions[nAction]:
        out[buttons.index(i)] = 1
    return out

observation, info = env. reset() #reset enviornment for new EPISODE

episode_over = False
total_reward = 0
start = time.time()
prev_ram = env.get_ram()

while not episode_over and time.time() - start <= 100:
    action_i = np.random.randint(len(actions))
    action = cnvrtAction(action_i) #will choose a random action from less possibilites
    # step is made, observe changes
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated
    ram = env.get_ram()
    diff = np.where(ram != prev_ram)[0]  # indices where bytes changed
    env.render() #show current frame
    print(diff)

env.close() #end
#Action Space: What can your agent do? (discrete choices, continuous values, etc.)
print(f"Action space: {env.action_space}")  # Discrete
print(f"Episode finished! Total reward: {total_reward}")



#firstbasic enviornment

#Observation Space: What can your agent see? (images, numbers, structured data, etc.)
