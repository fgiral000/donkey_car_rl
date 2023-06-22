import sys
import os

import gym_donkeycar 
import gym

from stable_baselines3 import SAC
from sb3_contrib import TQC


if __name__ == "__main__":


    #Se instancia el environment con ID
    env = gym.make("donkey-mountain-track-v0")

    env.reset()

    model = SAC('CnnPolicy', env, buffer_size=50000)

    model.learn(total_timesteps=500)

    env.close()