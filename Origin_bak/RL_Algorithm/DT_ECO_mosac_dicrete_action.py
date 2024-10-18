"""Multi-objective Soft Actor-Critic (SAC) algorithm for discrete action spaces.

It implements a multi-objective critic with weighted sum scalarization.
The implementation of this file is largely based on CleanRL's SAC implementation
https://github.com/vwxyzjn/cleanrl/blob/28fd178ca182bd83c75ed0d49d52e235ca6cdc88/cleanrl/sac_continuous_action.py
"""

import time
from copy import deepcopy
from typing import Optional, Tuple, Union
from typing_extensions import override

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import sys
import os
import csv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from morl_baselines.common.buffer import ReplayBuffer
from DT_ECO_Buffer import ReplayBuffer
from morl_baselines.common.evaluation import log_episode_info
# from morl_baselines.common.morl_algorithm import MOPolicy
from DT_ECO_Morl_algorithm import MOPolicy
from morl_baselines.common.networks import mlp, polyak_update
from morl_baselines.common.morl_algorithm import MOAgent

sys.path.append('/home/jiajiexu/DT_ECO/DataTrans/'); sys.path.append('/home/jiajiexu/DT_ECO/Model')
import TimingGraphTrans
import PhysicalDataTrans
import models
import dgl

# ALGO LOGIC: initialize agent here:
class MOSoftQNetwork(nn.Module):
    """Soft Q-network: S, A -> ... -> |R| (multi-objective)."""

    def __init__(
        self,
        obs_shape,
        action_dim,
        reward_dim,
    ):
        """Initialize the soft Q-network."""
        super().__init__()
        self.obs_shape = obs_shape # gate_sizes, timing_graph, layout, padding_mask
        self.action_dim = action_dim
        self.reward_dim = reward_dim

        # S, A -> ... -> |R| (multi-objective)
        # self.critic = mlp(
        #     input_dim=self.obs_shape,#这里不需要动作
        #     output_dim=self.action_dim * self.reward_dim,#输出是Q值和动作数量之积 softmax???
        #     net_arch=self.net_arch,
        #     activation_fn=nn.ReLU,
        # )
        self.critic = models.MultiModalNN(
            num_layers=3, 
            hidden_nf=64, 
            out_nf=32,
            output=self.action_dim*self.reward_dim, 
            embedding_dim=self.obs_shape['gate_sizes'][1],
            num_heads=4,
            num_Attention_layer=3,
            in_nf=self.obs_shape['timing_graph'][1], 
            in_ef=self.obs_shape['timing_graph'][4], 
            h1=32, 
            h2=32,
            in_channels=self.obs_shape['layout'][0]
        )

    def forward(self, obs):
        """Forward pass of the soft Q-network."""
        # Pass multi-modal inputs through the critic
        g = obs['timing_graph']
        img = obs['layout']
        padding_mask = obs['padding_mask']
        Gate_sizes = obs['gate_sizes']
        Gate_feature = obs['gate_features']
        
        # q_values = self.critic(g, img, padding_mask, Gate_feature, Gate_sizes)
        q_values = self.critic(g, img, padding_mask, Gate_sizes)
        print(f'q_values: {q_values}####{q_values.shape}')
        return q_values.view(-1, self.action_dim, self.reward_dim)



class MOSACActor(nn.Module):
    """Actor network: S -> A. Does not need any multi-objective concept."""

    def __init__(
        self,
        obs_shape: int,
        action_dim: int,
    ):
        """Initialize SAC actor."""
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        # print(f'obs_shape: {obs_shape}')

        # S -> ... -> |A| (mean)
        #          -> |A| (std)
        self.actor_net = models.MultiModalNN(
            num_layers=3, 
            hidden_nf=64, 
            out_nf=32,
            output=self.action_dim, 
            embedding_dim=self.obs_shape['gate_sizes'][1],
            num_heads=4,
            num_Attention_layer=3,
            in_nf=self.obs_shape['timing_graph'][1], 
            in_ef=self.obs_shape['timing_graph'][4],
            h1=32, 
            h2=32,
            in_channels=self.obs_shape['layout'][0]
        )


    def forward(self, obs):
        """Forward pass of the actor network."""
        g = obs['timing_graph']
        img = obs['layout']
        padding_mask = obs['padding_mask']
        Gate_sizes = obs['gate_sizes']
        Gate_feature = obs['gate_features']
        logits = self.actor_net(g, img, padding_mask, Gate_sizes)

        return logits

    def get_action(self, obs):
        """Get action from the actor network."""
        logits = self(obs)
        action_probs = F.softmax(logits, dim = -1) # action probability distribution
        action_dist = th.distributions.Categorical(action_probs)
        action = action_dist.sample().view(-1, 1) # sample action from distribution
        z = (action_probs == 0.0).float() * 1e-8 # prevent instability
        log_probs = th.log(action_probs + z)
        #print(f'log_probs: {log_probs.shape}')
        #print(f'action.shape: {action.shape}')
        #print(f'action_probs.shape: {action_probs.shape}')
        return action, log_probs, action_probs


def GPU_Memory_Monitor():
    if th.cuda.is_available():
        device = th.device('cuda')

        # already allocated memory
        allocated_memory = th.cuda.memory_allocated(device)

        # reserved memory
        reserved_memory = th.cuda.memory_reserved(device)

        print(f"Allocated memory: {allocated_memory / (1024**2):.2f} MB")
        print(f"Reserved memory: {reserved_memory / (1024**2):.2f} MB")
    else:
        print("No GPU available.")

class MOSAC(MOPolicy, MOAgent):
    """Multi-objective Soft Actor-Critic (SAC) algorithm.

    It is a multi-objective version of the SAC algorithm, with multi-objective critic and weighted sum scalarization.
    """

    def __init__(
        self,
        env: gym.Env,
        weights: np.ndarray,
        scalarization=th.matmul,
        buffer_size: int = int(1e6),
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        learning_starts: int = int(1e3),
        policy_lr: float = 3e-4,
        q_lr: float = 1e-4,
        a_lr: float = 1e-4,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "MOSAC",
        policy_freq: int = 2,
        target_net_freq: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,
        id: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        log_every: int = 1000,
        seed: int = 42,
        parent_rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the MOSAC algorithm.

        Args:
            env: Env
            weights: weights for the scalarization
            scalarization: scalarization function
            buffer_size: buffer size
            gamma: discount factor
            tau: target smoothing coefficient (polyak update)
            batch_size: batch size
            learning_starts: how many steps to collect before triggering the learning
            policy_lr: learning rate of the policy
            q_lr: learning rate of the q networks
            policy_freq: the frequency of training policy (delayed)
            target_net_freq: the frequency of updates for the target networks
            alpha: Entropy regularization coefficient
            autotune: automatic tuning of alpha
            id: id of the SAC policy, for multi-policy algos
            device: torch device
            torch_deterministic: whether to use deterministic version of pytorch
            log: logging activated or not
            seed: seed for the random generators
            parent_rng: parent random generator, for multi-policy algos
        """
        MOAgent.__init__(self, env, device, seed=seed)
        MOPolicy.__init__(self, None, device)
        # Seeding
        self.seed = seed
        self.parent_rng = parent_rng
        if parent_rng is not None:
            self.np_random = parent_rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        # env setup
        self.env = env
        #assert isinstance(self.env.action_space, gym.spaces.Box), "only continuous action space is supported"
        self.obs_shape = self.observation_shape
        self.action_dim = self.action_dim
        self.reward_dim = self.reward_dim
        # print(f'self.obs_shape: {self.obs_shape}')
        # print(f'self.action_dim: {self.action_dim}')
        # print(f'self.reward_dim: {self.reward_dim}')

        # Scalarization
        self.weights = weights
        self.weights_tensor = th.from_numpy(self.weights).float().to(self.device)
        self.batch_size = batch_size
        self.scalarization = scalarization

        # SAC Parameters
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.learning_starts = learning_starts
        self.policy_lr = policy_lr
        self.learning_rate = policy_lr
        self.q_lr = q_lr
        self.a_lr = a_lr
        self.policy_freq = policy_freq
        self.target_net_freq = target_net_freq

        # Networks
        self.actor = MOSACActor(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
        ).to(self.device)# input observation space, output action

        
        self.qf1 = MOSoftQNetwork(
            obs_shape=self.obs_shape, action_dim=self.action_dim, reward_dim=self.reward_dim
        ).to(self.device)# input observation space, output action dim * reward dim
        
        
        self.qf2 = MOSoftQNetwork(
            obs_shape=self.obs_shape, action_dim=self.action_dim, reward_dim=self.reward_dim
        ).to(self.device)
        
        
        self.qf1_target = MOSoftQNetwork(
            obs_shape=self.obs_shape, action_dim=self.action_dim, reward_dim=self.reward_dim
        ).to(self.device)
        
        
        self.qf2_target = MOSoftQNetwork(
            obs_shape=self.obs_shape, action_dim=self.action_dim, reward_dim=self.reward_dim
        ).to(self.device)
        
        
        self.qf1_target.requires_grad_(False)
        self.qf2_target.requires_grad_(False)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr)

        
        # Automatic entropy tuning
        self.autotune = autotune
        if self.autotune:
            #self.target_entropy =  -np.log(1/self.action_dim)
            self.target_entropy =  0.98*-np.log(1.0/self.action_dim)
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            #self.log_alpha = th.tensor([-1.0], requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.a_lr)
        else:
            self.alpha = alpha
        self.alpha_tensor = th.tensor([self.alpha]).to(self.device)
        
        # Buffer
        # self.env.observation_space.dtype = np.float32
        
        self.buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            action_dim=self.action_shape[0],
            rew_dim=self.reward_dim,
            max_size=self.buffer_size,
            device=self.device
        )

        
        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        self.log_every = log_every
        if log and parent_rng is None:
            self.setup_wandb(self.project_name, self.experiment_name, wandb_entity)

    def get_config(self) -> dict:
        """Returns the configuration of the policy."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "policy_lr": self.policy_lr,
            "q_lr": self.q_lr,
            "policy_freq": self.policy_freq,
            "target_net_freq": self.target_net_freq,
            "alpha": self.alpha,
            "autotune": self.autotune,
            "seed": self.seed,
        }

    def __deepcopy__(self, memo):
        """Deep copy of the policy.

        Args:
            memo (dict): memoization dict
        """
        copied = type(self)(
            env=self.env,
            weights=self.weights,
            scalarization=self.scalarization,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            tau=self.tau,
            batch_size=self.batch_size,
            learning_starts=self.learning_starts,
            policy_lr=self.policy_lr,
            q_lr=self.q_lr,
            a_lr=self.a_lr,
            policy_freq=self.policy_freq,
            target_net_freq=self.target_net_freq,
            alpha=self.alpha,
            autotune=self.autotune,
            id=self.id,
            device=self.device,
            log=self.log,
            seed=self.seed,
            parent_rng=self.parent_rng,
            project_name = self.project_name,
            experiment_name = self.experiment_name
        )

        # Copying networks
        copied.actor = deepcopy(self.actor)
        copied.qf1 = deepcopy(self.qf1)
        copied.qf2 = deepcopy(self.qf2)
        copied.qf1_target = deepcopy(self.qf1_target)
        copied.qf2_target = deepcopy(self.qf2_target)

        copied.global_step = self.global_step
        copied.actor_optimizer = optim.Adam(copied.actor.parameters(), lr=self.policy_lr, eps=1e-5)
        copied.q_optimizer = optim.Adam(list(copied.qf1.parameters()) + list(copied.qf2.parameters()), lr=self.q_lr)
        if self.autotune:
            copied.a_optimizer = optim.Adam([copied.log_alpha], lr=self.a_lr)
        copied.alpha_tensor = th.scalar_tensor(copied.alpha).to(self.device)
        copied.buffer = deepcopy(self.buffer)
        return copied

    @override
    def get_buffer(self):
        return self.buffer

    @override
    def set_buffer(self, buffer):
        self.buffer = buffer

    @override
    def get_policy_net(self) -> th.nn.Module:
        return self.actor

    @override
    def set_weights(self, weights: np.ndarray):
        self.weights = weights
        self.weights_tensor = th.from_numpy(self.weights).float().to(self.device)

    @override
    def eval(self, obs: dict, w: Optional[np.ndarray] = None) -> Union[int, np.ndarray]:
        """Returns the best action to perform for the given obs.

        Args:
            obs: observation as a numpy array
            w: None
        Return:
            action as a numpy array (continuous actions)
        """
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs[key] = th.as_tensor(value, dtype=th.float32).to(self.device).unsqueeze(0)
            elif isinstance(value, th.Tensor):
                obs[key] = value.to(self.device).unsqueeze(0)
            else:
                obs[key] = value.to(self.device)
        with th.no_grad():
            action, _, _ = self.actor.get_action(obs)

        return action[0].detach().cpu().numpy()

    @override
    def update(self):
        (mb_obs, mb_act, mb_rewards, mb_next_obs, mb_dones) = self.buffer.sample(
            self.batch_size, device=self.device, use_cer=True
        )# sample a batch of experiences from the buffer
        # print('mb_obs',mb_obs['timing_graph'].ndata['feature'].shape)
        
        with th.no_grad():
            _, log_probs, action_probs = self.actor.get_action(mb_next_obs)
            # (!) Q values are scalarized before being compared (min of ensemble networks)
            # print('mb_next_obs',mb_next_obs['timing_graph'].ndata['feature'].shape)
            #print(f'self.qf1_target(mb_next_obs): {self.qf1_target(mb_next_obs)}####{self.qf1_target(mb_next_obs).shape}')#64,14,2
            #print(f'self.weights_tensor: {self.weights_tensor.shape}')
            qf1_next_target = (self.qf1_target(mb_next_obs) * self.weights_tensor).sum(dim = -1) #(batch size, action dim, reward dim) (1, 1, reward dim) calculate the  total reward
            qf2_next_target = (self.qf2_target(mb_next_obs) * self.weights_tensor).sum(dim = -1) 
            
            #print(f'qf1_next_target: {qf1_next_target}####{qf1_next_target.shape}')
            #print(f'qf2_next_target: {qf2_next_target}####{qf2_next_target.shape}')
            
            soft_state_values = (action_probs * (th.min(qf1_next_target, qf2_next_target) - self.alpha_tensor * log_probs)).sum(dim = 1)#(128,9)
            scalarized_rewards = (mb_rewards * self.weights_tensor).sum(dim = -1)
            next_q_value = scalarized_rewards.flatten() + (~mb_dones.flatten()).float() * self.gamma * soft_state_values

        qf1_a_values = (self.qf1(mb_obs) * self.weights_tensor).sum(dim = -1).gather(1, mb_act.long()).squeeze(-1)
        qf2_a_values = (self.qf2(mb_obs) * self.weights_tensor).sum(dim = -1).gather(1, mb_act.long()).squeeze(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)# calculate mse between qf1_a_values and next_q_value
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        self.q_optimizer.step()
        # GPU_Memory_Monitor()
        
        if self.global_step % self.policy_freq == 0:  # TD 3 Delayed update support
            for _ in range(self.policy_freq):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                _, log_probs, action_probs = self.actor.get_action(mb_obs)
                entropies = th.sum(action_probs * log_probs, dim=1)
                # (!) Q values are scalarized before being compared (min of ensemble networks)
                qf1_values = (self.qf1(mb_obs) * self.weights_tensor).sum(dim = -1)
                qf2_values = (self.qf2(mb_obs) * self.weights_tensor).sum(dim = -1)
                min_qf_values = th.min(qf1_values, qf2_values)
                actor_loss = (action_probs * (self.alpha_tensor * log_probs - min_qf_values)).sum(dim = 1).mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune: # automatic tuning of alpha
                    alpha_loss = (self.log_alpha * (self.target_entropy + entropies).detach()).mean()

                    self.a_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha_tensor = self.log_alpha.exp()
                    self.alpha = max(self.log_alpha.exp().item(), 0.01)

        # update the target networks
        if self.global_step % self.target_net_freq == 0:
            polyak_update(params=self.qf1.parameters(), target_params=self.qf1_target.parameters(), tau=self.tau)
            polyak_update(params=self.qf2.parameters(), target_params=self.qf2_target.parameters(), tau=self.tau)
            self.qf1_target.requires_grad_(False)
            self.qf2_target.requires_grad_(False)
        
        if self.global_step % 100 == 0 and self.log:
            log_str = f"_{self.id}" if self.id is not None else ""
            to_log = {
                f"losses{log_str}/alpha": self.alpha,
                f"losses{log_str}/qf1_values": qf1_a_values.mean().item(),
                f"losses{log_str}/qf2_values": qf2_a_values.mean().item(),
                f"losses{log_str}/qf1_loss": qf1_loss.item(),
                f"losses{log_str}/qf2_loss": qf2_loss.item(),
                f"losses{log_str}/actor_loss": actor_loss.item(),
                "global_step": self.global_step,
            }
            if self.autotune:
                to_log[f"losses{log_str}/alpha_loss"] = alpha_loss.item()
            wandb.log(to_log)
        
    def train(self, total_timesteps: int, eval_env: Optional[gym.Env] = None, start_time=None):
        """Train the agent.

        Args:
            total_timesteps (int): Total number of timesteps (env steps) to train for
            eval_env (Optional[gym.Env]): Gym environment used for evaluation.
            start_time (Optional[float]): Starting time for the training procedure. If None, it will be set to the current time.
        """
        if start_time is None:
            start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.env.reset()
        for step in range(total_timesteps):
            # ALGO LOGIC: put action logic here
            if self.global_step < self.learning_starts:
                actions = self.env.action_space.sample()
            else:
                th_obs = {}
                for key, value in obs.items():
                    if isinstance(value, th.Tensor):
                        th_obs[key] = value.to(self.device).unsqueeze(0)
                    else:
                        th_obs[key] = value.to(self.device)
                actions, _, _ = self.actor.get_action(th_obs)
                actions = actions.detach().cpu().numpy()

            # execute the game and log data
            if actions.ndim == 2:
                actions = np.squeeze(actions)
            next_obs, rewards, terminated, allterminated, infos = self.env.step(actions)
            print(f'Timing Step: {step}, rewards: {rewards}')

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs
            if "final_observation" in infos:
                real_next_obs = infos["final_observation"]
            self.buffer.add(obs=obs, next_obs=real_next_obs, action=actions, reward=rewards, done=allterminated)
            #print(rewards)
            if terminated == True:
                # Log rewards to a CSV file 
                # print(rewards)
                rewards_to_log = rewards # Convert tensor to numpy array for easier handling 
                csv_filename = f"rewards.csv" 
                file_exists = os.path.isfile(csv_filename) 

                with open(csv_filename, mode='a', newline='') as file: 
                    writer = csv.writer(file) 
                    if not file_exists: 
                        writer.writerow(["Reward[0]", "Reward[1]"]) # Write header if file doesn't exist 
                    writer.writerow([rewards_to_log[0], rewards_to_log[1]]) 
                log_str = f"_{self.id}" if self.id is not None else ""
                to_log = {f"rewards{log_str}/Reward[0]": rewards_to_log[0]}
                wandb.log(to_log)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            if allterminated:
                obs, _ = self.env.reset()
                if self.log and "episode" in infos.keys():
                    log_episode_info(infos["episode"], np.dot, self.weights, self.global_step, self.id)

            # ALGO LOGIC: training.
            if self.global_step > self.learning_starts:
                self.update()
                # print(self.global_step)
                if self.log and self.global_step % 100 == 0:
                    print('SecondPerStep:', float(self.global_step / (time.time() - start_time)))
                    print('Time consumed:', time.time() - start_time)
                    wandb.log(
                        {"charts/SPS": float(self.global_step / (time.time() - start_time)), "global_step": self.global_step}
                    )

            self.global_step += 1
