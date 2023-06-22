import gym
from gym.wrappers import NormalizeObservation, TimeLimit
import gym_donkeycar

import stable_baselines3
from stable_baselines3 import SAC
from sb3_contrib import TQC 

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from callbacks_from_rlzoo import ParallelTrainCallback
from wrappers_from_rlzoo import HistoryWrapper

from ae.wrapper import AutoencoderWrapper



if __name__ == "__main__":


    ######### ENVIRONMENT ##########
    try:
        #We define the environment from the donkey car repository

        conf_car = {"car_config": {"body_style": "f1",
                                    "body_rgb": (63, 165, 157),
                                    "car_name": "FranGiral",
                                    "font_size": 16}}

        # env = gym.make("donkey-mountain-track-v0", conf = conf_car)
        env = gym.make("donkey-minimonaco-track-v0", conf = conf_car)

        # # env = DummyVecEnv([lambda: env])
        #Now we wrapp the environment with the different wrappers we want
        env = Monitor(env)
        env = TimeLimit(env, max_episode_steps= 2000)
        env = AutoencoderWrapper(env)
        env = NormalizeObservation(env)##########################################
        env = HistoryWrapper(env, horizon=5)
        env.reset()


        #Now we define the callback for parallel training 
        # parallel_callback = ParallelTrainCallback(gradient_steps=200)


        ######### AGENT ###########

        #We create the agent that we want to train using sb3
        #We are going to use TQC agent
        MODEL_PATH = "first_donkey_monaco_tqc_700k.zip"
        tqc_model = TQC.load(MODEL_PATH, env)
        # tqc_model.load_replay_buffer("tqc-monaco-replay-buffer-cte12-700k.pkl")
        
        
        tqc_model.set_parameters(load_path_or_dict=MODEL_PATH)
        tqc_model.set_env(env=env)
        policy = tqc_model.policy
        mean, std = evaluate_policy(tqc_model, env, n_eval_episodes=40, deterministic=True)
        print(mean)
        print(std)

        # observation = env.reset()
        # #Se inicializa una lista para almacenar los rewards de un episodio
        # ep_returns = []


        # for _ in range(10):
        #     observation = env.reset()
        #     done = False
        #     rewards = []
        #     while not done:
        #         action, _ = tqc_model.predict(observation, deterministic=True) # Tomar una acción aleatoria
        #         new_observation, reward, done, _ = env.step(action)
        #         observation = new_observation
        #         rewards.append(reward)
        #         print(reward)

    except KeyboardInterrupt:
        env.close()