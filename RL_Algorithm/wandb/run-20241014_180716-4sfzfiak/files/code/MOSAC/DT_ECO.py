import mo_gymnasium as mo_gym
# import gymnasium as gym
import numpy as np
from mosac_dicrete_action import MOSAC


env = mo_gym.make('dt-eco-v0')

# eval_env = mo_gym.make('dt-eco-v0')

# env = mo_gym.make('dim-test-v0')

# eval_env = mo_gym.make('dim-test-v0')

#'''
agent = MOSAC(
    env=env,
    weights = np.array([0.5,0.5]), # weights of reward values
    #scalarization = th.matmul, # 使用矩阵乘法作为默认的标量化操作 
    buffer_size = int(100), # size of resampling buffer 
    gamma = 0.99, # discount factor
    tau = 0.005, # soft update parameter
    batch_size = 5, # batch size
    learning_starts = int(0), # random sampling befero learning
    policy_lr = 1e-4, # learning rate of policy network
    q_lr = 1e-4, # learning rate of Q network
    a_lr = 1e-4,
    policy_freq = 2, # update policy network frequency
    target_net_freq = 1, # update target network frequency
    alpha = 0.1, # Entropy term coefficient
    autotune = True, # random tuning alpha 
    log = True, # record log or not 
    seed = 42, # random seed

)

obs, _ = env.reset()
# for i in range(100):
#     env.render()
#     actions = env.action_space.sample()
#     next_obs, rewards, terminated, truncated, infos = env.step(actions)
#     real_next_obs = next_obs
#     agent.buffer.add(obs=obs, next_obs=real_next_obs, action=actions, reward=rewards, done=terminated)
#     obs = next_obs
#     print(i)
    
# (mb_obs, mb_act, mb_rewards, mb_next_obs, mb_dones) = agent.buffer.sample(64)
# print(mb_obs['gate_sizes'])
agent.train(
    total_timesteps=100,
    eval_env=env,
    )
# '''