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

env_name = "humanoid"  # @param ['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'ur5e', 'reacher']
env_fn = envs.create_fn(env_name=env_name)
env = env_fn()
jit_env_reset = jax.jit(env.reset)
state = jit_env_reset(rng=jax.random.PRNGKey(seed=0))

keywords = {
num_timesteps: int(3e6),
log_frequency: 10000,
reward_scaling: 100,
episode_length: 1000,
normalize_observations: True,
action_repeat: 1, 
discounting: 0.99,
learning_rate: 4.5e-4, 
num_envs: 128, 
batch_size: 512,
min_replay_size: 8192,
max_replay_size: int(3e6),
grad_updates_per_step: 0.25, 
max_devices_per_host: 8
}
 
train_fn = functools.partial(
              sac.train, **keywords)
              #num_timesteps = 1048576 * 2.2,
                    #log_frequency = 20000, reward_scaling = 100, episode_length = 1000,
                     #     normalize_observations = True, action_repeat = 1, discounting = 0.99,
                     #           learning_rate = 4.5e-4, num_envs = 128, batch_size = 512,
                      #                min_replay_size = 8192, max_replay_size = 1048576,
                       #                 grad_updates_per_step = 0.25, max_devices_per_host = 8
                        #                    )
max_y = {'ant': 6000, 
                 'humanoid': 12000, 
                          'fetch': 15, 
                                   'grasp': 100, 
                                            'halfcheetah': 8000,
                                                     'ur5e': 10,
                                                              'reacher': 5}[env_name]

min_y = {'reacher': -100}.get(env_name, 0)
import wandb
wandb.init(project='brax')
xdata = []
ydata = []
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

file = html.render(env.sys, state.qps) 
with open("final_agent.html", "w") as f: 
    f.write(file)
wandb.log({'final_agent': wandb.Html("final_agent.html")})
#print(f'time to jit: {times[1] - times[0]}')
#print(f'time to train: {times[-1] - times[1]}')
