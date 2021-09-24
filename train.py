from datetime import datetime
import functools
import os

#from IPython.display import HTML, clear_output
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import brax

from brax import envs
from brax.training import ppo, sac
from brax.io import html
import wandb 

env_name = "humanoid"  # @param ['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'ur5e', 'reacher']
env_fn = envs.create_fn(env_name=env_name)
env = env_fn()
jit_env_reset = jax.jit(env.reset)
state = jit_env_reset(rng=jax.random.PRNGKey(seed=0))

#keywords = {
#num_timesteps: int(3e6),
#log_frequency: 10000,
#reward_scaling: 100,
#episode_length: 1000,
#normalize_observations: True,
#action_repeat: 1, 
#discounting: 0.99,
#learning_rate: 4.5e-4, 
#num_envs: 128, 
#batch_size: 512,
#min_replay_size: 8192,
#max_replay_size: int(3e6),
#grad_updates_per_step: 0.25, 
#max_devices_per_host: 8
#}
#for lr in [3e-4, 4e-4, 3.5e-4, 4.5e-4, 5e-4, 5.5e-4, 6e-4]:
lr = 5.5e-4
#for dis in [0.94, 0.95, 0.96, 0.97, 0.98]:
for dis in [0.982, 0.984, 0.986, 0.988, 0.992, 0.994, 0.996, 0.998]:
    for seed in [0, 1, 3, 5, 7, 9]:
        keywords={
        'seed': seed,
    'num_timesteps': int(5e6),
    'log_frequency': 50000,
    'reward_scaling': 100,
    'episode_length': 1000,
    'normalize_observations': True,
    'action_repeat': 1,
    'discounting': dis,
    'learning_rate': lr,
    'num_envs': 256,
    'batch_size': 512,
    'min_replay_size': 102400,
    'max_replay_size': 1024000,
    'grad_updates_per_step': 0.25,
    'max_devices_per_host': 8
    }
        train_fn = functools.partial(
              sac.train, **keywords)
              #num_timesteps = 1048576 * 2.2,
                    #log_frequency = 20000, reward_scaling = 100, episode_length = 1000,
                     #     normalize_observations = True, action_repeat = 1, discountin
        max_y = {'ant': 6000, 
                 'humanoid': 12000, 
                          'fetch': 15, 
                                   'grasp': 100, 
                                            'halfcheetah': 8000,
                                                     'ur5e': 10,
                                                              'reacher': 5}[env_name]

        min_y = {'reacher': -100}.get(env_name, 0)
        run = wandb.init(project='brax')
        run.config.update(keywords)
#xdata = []
#ydata = []
        times = [datetime.now()]
        def progress(num_steps, metrics):
            times.append(datetime.now())
        #print(datetime.now(), num_steps)
            metrics.update({'# environment steps': num_steps} )
            wandb.log(metrics)
            return 


        inference_fn, params, _ = train_fn(environment_fn=env_fn, progress_fn=progress)


        jit_env_step = jax.jit(env.step)
        jit_inference_fn = jax.jit(inference_fn)
        rng = jax.random.PRNGKey(seed=0)
        reset_key, rng = jax.random.split(rng)
        state = jit_env_reset(rng=reset_key)
        qps = []
        while not state.done:
            qps.append(state.qp)
            tmp_key, rng = jax.random.split(rng)
            act = jit_inference_fn(params, state.obs, tmp_key)
            state = jit_env_step(state, act)

        file = html.render(env.sys, qps) 
        with open("final_agent.html", "w") as f: 
            f.write(file)
        wandb.log({'final_agent': wandb.Html("final_agent.html")})
        wandb.finish()
        del inference_fn 
        del params
#print(f'time to jit: {times[1] - times[0]}')
#print(f'time to train: {times[-1] - times[1]}')
