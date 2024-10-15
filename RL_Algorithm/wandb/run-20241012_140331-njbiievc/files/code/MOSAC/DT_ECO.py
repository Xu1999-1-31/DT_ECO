import mo_gymnasium as mo_gym
# import gymnasium as gym
import numpy as np
from mosac_dicrete_action import MOSAC


env = mo_gym.make('dt-eco-v0')

eval_env = mo_gym.make('dt-eco-v0')

# env = mo_gym.make('dim-test-v0')

# eval_env = mo_gym.make('dim-test-v0')

#'''
agent = MOSAC(
    env=env,
    weights = np.array([0.5,0.5]), # 假设某种权重数组，具体可根据需求调整 
    #scalarization = th.matmul, # 使用矩阵乘法作为默认的标量化操作 
    buffer_size = int(1000), # 回放缓冲区的大小 
    gamma = 0.99, # 折扣因子 
    tau = 0.005, # 软更新参数 
    batch_size = 128, # 批次大小 
    learning_starts = int(0), # 训练开始的时间步数 
    policy_lr = 1e-4, # 策略网络的学习率 
    q_lr = 1e-4, # Q网络的学习率 
    a_lr = 1e-4,
    policy_freq = 2, # 更新策略网络的频率 
    target_net_freq = 1, # 更新目标网络的频率 
    alpha = 0.1, # 温度参数 
    autotune = True, # 是否自动调整alpha 
    log = True, # 是否记录日志 
    seed = 42, # 随机种子

)

obs, _ = env.reset()
for _ in range(10000):
    env.render()
    actions = env.action_space.sample()
    next_obs, rewards, terminated, truncated, infos = env.step(actions)
    real_next_obs = next_obs
    agent.buffer.add(obs=obs, next_obs=real_next_obs, action=actions, reward=rewards, done=terminated)
    obs = next_obs
    
(mb_obs, mb_act, mb_rewards, mb_next_obs, mb_dones) = agent.buffer.sample(128)
print(mb_obs)
# agent.train(
#     total_timesteps=100,
#     eval_env=eval_env,
#     )
# '''