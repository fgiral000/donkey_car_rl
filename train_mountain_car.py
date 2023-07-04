import gym
# from gym.wrappers import NormalizeObservation
from gym.wrappers import TimeLimit
import gym_donkeycar

import stable_baselines3
from stable_baselines3 import SAC
from sb3_contrib import TQC 

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from callbacks_from_rlzoo import ParallelTrainCallback, LapTimeCallback
from wrappers_from_rlzoo import HistoryWrapper, NormalizeObservation

from ae.wrapper import AutoencoderWrapper

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize





if __name__ == "__main__":


    ######### ENVIRONMENT ##########

    #We define the environment from the donkey car repository

    conf_car = {"car_config": {"body_style": "f1",
                                "body_rgb": (63, 165, 157),
                                "car_name": "FranGiral",
                                "font_size": 16}}

    # env = gym.make("donkey-mountain-track-v0", conf = conf_car)
    env = gym.make("donkey-minimonaco-track-v0", conf = conf_car)

    #Now we wrapp the environment with the different wrappers we want
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps= 2000)
    env = AutoencoderWrapper(env)
    # env = NormalizeObservation(env)
    env = HistoryWrapper(env, horizon=5)

    #VecNormalize wrappers
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env,
    #                    training=True,
    #                    norm_obs=True,
    #                    norm_reward=False,
    #                    clip_obs=10)
    env = VecNormalize.load(load_path="vec_normalize.pkl", venv= env)

    # env.reset()

    #EVALUATION ENV
    # eval_env = gym.make("donkey-minimonaco-track-v0", conf = conf_car)
    # eval_env = Monitor(eval_env)
    # eval_env = TimeLimit(eval_env, max_episode_steps= 2000)
    # eval_env = AutoencoderWrapper(eval_env)
    # eval_env = NormalizeObservation(eval_env)
    # eval_env = HistoryWrapper(eval_env, horizon=5)
    # eval_env.reset()





    lap_time_callback = LapTimeCallback()
    ## Weights and biases integration with sb3
    run = wandb.init(
                    name='TQC-test-mean-std',
                    project="racing_rl",
                    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                    monitor_gym=True,  # auto-upload the videos of agents playing the game
                    save_code=True,  # optional
                        )
    
    #We define the WANDB callback
    wandb_callback = WandbCallback(verbose=2, model_save_path=f"models/{run.id}", model_save_freq=10000)
    
    #callback for evaluation during training
    # eval_callback = EvalCallback(eval_env=eval_env, eval_freq=5000, best_model_save_path=f"models/{run.id}/best_model")



    #Now we define the callback for parallel training 
    parallel_callback = ParallelTrainCallback(gradient_steps=200)
    # checkpoint_callback = CheckpointCallback(save_freq = 10000, save_path="logs/model-checkpoint-mountain",name_prefix="training_test")
    
    ######### AGENT ###########

    #We create the agent that we want to train using sb3
    #We are going to use TQC agent

    # tqc_model = TQC("MlpPolicy",
    #                 env=env,
    #                 learning_rate=7.3e-4,
    #                 buffer_size=20000,
    #                 batch_size=256,
    #                 ent_coef='auto',
    #                 gamma=0.99,
    #                 tau=0.02,
    #                 train_freq=200,
    #                 gradient_steps=256,
    #                 learning_starts=500,
    #                 use_sde_at_warmup=True,
    #                 use_sde=True,
    #                 sde_sample_freq=16,
    #                 policy_kwargs=dict(log_std_init=-3, net_arch=[256,256], n_critics = 2),
    #                 tensorboard_log=f"runs/{run.id}",
    #                 verbose = 2,
    #                 seed = 45554)
    

    ###### TRAINING #########

    #####Un-comment when you want to train from a pre-trained model
    tqc_model = TQC.load("first_donkey_monaco_tqc_VecEnv", custom_objects={"verbose":2, "tensorboard_log": f"runs/{run.id}", "learning_starts":0})
    # tqc_model = TQC.load("first_donkey_mountain_tqc_415k.zip")
    tqc_model.load_replay_buffer("tqc-monaco-replay-buffer-VecEnv.pkl")
    tqc_model.set_env(env=env)

    #Train the model
    tqc_model._last_obs = None
    tqc_model.learn(total_timesteps=100e3, callback=[parallel_callback, lap_time_callback, wandb_callback])

    #Save the model
    tqc_model.save("first_donkey_monaco_tqc_VecEnv-2")
    tqc_model.save_replay_buffer("tqc-monaco-replay-buffer-VecEnv-2")
    env.save("vec_normalize_2.pkl")


    # policy = tqc_model.policy
    # mean, std = evaluate_policy(tqc_model, env, n_eval_episodes=4, deterministic=True)
    # print(mean)
    # print(std)


    # env.close()
    run.finish()