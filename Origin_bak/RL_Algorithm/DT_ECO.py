import numpy as np
from DT_ECO_mosac_dicrete_action import MOSAC

#'''
def create_agent(env):
    agent = MOSAC(
        env=env,
        weights = np.array([1, 0]), # weights of reward values
        buffer_size = int(1000), # size of resampling buffer 
        gamma = 0.99, # discount factor
        tau = 0.005, # soft update parameter
        batch_size = 64, # batch size
        learning_starts = int(1000), # random sampling before learning
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
    return agent

# '''
