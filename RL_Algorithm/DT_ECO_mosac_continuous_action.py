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

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var
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
        net_arch=[256, 256],
    ):
        """Initialize the soft Q-network."""
        super().__init__()
        self.obs_shape = obs_shape # delta_strength, timing graph
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        # S, A -> ... -> |R| (multi-objective)
        self.GNN = models.MultiLayerTimingGNN(3, 32)
        
        self.critic = mlp(
            input_dim=32 + np.prod(self.action_dim),
            output_dim=self.reward_dim,
            net_arch=self.net_arch,
            activation_fn=nn.ReLU,
        )
        
    def forward(self, obs, a):
        """Forward pass of the soft Q-network."""
        g = obs['timing_graph']
        with g.local_scope():
            g.ndata['nf'] = self.GNN(g)
            nf = dgl.mean_nodes(g, 'nf') # batch size, feature dim
        a = a.squeeze(-1)
        x = th.cat([nf, a], dim=1)
        
        # q_values = self.critic(g, img, padding_mask, Gate_feature, Gate_sizes)
        q_values = self.critic(x)
        return q_values

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class MOSACActor(nn.Module):
    """Actor network: S -> A. Does not need any multi-objective concept."""

    def __init__(
        self,
        obs_shape,
        action_dim,
        reward_dim,
        action_lower_bound,
        action_upper_bound,
        libcells = 810,
        net_arch=[256, 128, 64],
    ):
        """Initialize SAC actor."""
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.libcells = libcells
        self.net_arch = net_arch
        # print(f'obs_shape: {obs_shape}')

        # S -> ... -> |A| (mean)
        #          -> |A| (std)
        self.GNN = models.MultiLayerTimingGNN(3, 32)
        
        self.register_buffer("action_scale", th.tensor((action_upper_bound - action_lower_bound) / 2.0, dtype=th.float32))
        self.register_buffer("action_bias", th.tensor((action_upper_bound + action_lower_bound) / 2.0, dtype=th.float32))
        
        self.Embedding = th.nn.Embedding(num_embeddings=self.libcells, embedding_dim=32)
        self.encoder = models.LSTMEncoder(np.prod(self.action_dim), 64, 2) # in dim, hidden dim
        self.decoder_mean = models.SelfAttentionDecoder(64, 64, 4, 3) # hidden dim, out dim, num head, num layer
        self.decoder_logstd = models.SelfAttentionDecoder(64, 64, 4, 3) # hidden dim, out dim, num head, num layer
        self.mlp_mean = mlp(
            input_dim=64,
            output_dim=1,
            net_arch=self.net_arch,
            activation_fn=nn.ReLU,
        )
        self.mlp_logstd = mlp(
            input_dim=64,
            output_dim=1,
            net_arch=self.net_arch,
            activation_fn=nn.ReLU,
        )


    def forward(self, obs):
        """Forward pass of the actor network."""
        delta_strength = obs['delta_strength']
        g = obs['timing_graph']
        with g.local_scope():
            g.ndata['nf'] = self.GNN(g)
            mask = g.ndata['padding_mask']
            nf = g.ndata['nf'][mask]
            celltype = g.ndata['celltype'][mask] # batch * n nodes, feature dim
        embedding = self.Embedding(celltype) # batch * n, feature dim, 32
        embedding = embedding.squeeze(1)
        cellfeature = th.cat([nf, embedding], dim=-1)
        if delta_strength.dim() == 3:  # batched, (batch_size, seqlen, feature_dim) 10 here
            batch_size, seqlen, delta_feature_dim = delta_strength.shape
        elif delta_strength.dim() == 2:  # unbatched, (seqlen, feature_dim)
            batch_size = 1
            seqlen, delta_feature_dim = delta_strength.shape
            delta_strength = delta_strength.unsqueeze(0)
        node_feature_dim = cellfeature.size(-1)
        padded_cellfeatures = th.zeros(batch_size, delta_feature_dim, node_feature_dim).to(cellfeature.device) # cells length == delta feature dim
        node_offset = 0
        for i in range(batch_size):
            node_features = cellfeature[node_offset:node_offset + delta_feature_dim]  # batch nodes
            num_nodes = node_features.size(0)
            padded_cellfeatures[i, :num_nodes, :] = node_features
            node_offset += num_nodes
        query, _, _ = self.encoder(delta_strength)
        action_mean = self.decoder_mean(query, padded_cellfeatures)
        action_logstd = self.decoder_logstd(query, padded_cellfeatures)
        action_mean = self.mlp_mean(action_mean) # batch size, delta feature dim, 1
        action_logstd = self.mlp_logstd(action_logstd)
        action_logstd = th.tanh(action_logstd)
        action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)
        return action_mean, action_logstd

    def get_action(self, x):
        """Get action from the actor network."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = th.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = th.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = th.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


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
        q_lr: float = 1e-3,
        # a_lr: float = 1e-4,
        project_name: str = "DT_ECO",
        experiment_name: str = "dt_eco1",
        policy_freq: int = 2,
        target_net_freq: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,
        id: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        log_every: int = 10,
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
        assert isinstance(self.env.action_space, gym.spaces.Box), "only continuous action space is supported"
        self.obs_shape = self.observation_shape
        self.action_dim = self.action_shape
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
        # self.a_lr = a_lr
        self.policy_freq = policy_freq
        self.target_net_freq = target_net_freq

        # Networks
        self.actor = MOSACActor(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            action_lower_bound=self.env.action_space.low,
            action_upper_bound=self.env.action_space.high,
        ).to(self.device)# input observation space, output action

        
        self.qf1 = MOSoftQNetwork(
            obs_shape=self.obs_shape, action_dim=self.action_dim, reward_dim=self.reward_dim
        ).to(self.device)# input observation space, output reward dim
        
        
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
            self.target_entropy = -th.prod(th.Tensor(env.action_space.shape).to(self.device)).item()
            # self.target_entropy =  0.98*-np.log(1.0/self.action_dim)
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            #self.log_alpha = th.tensor([-1.0], requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr)
        else:
            self.alpha = alpha
        self.alpha_tensor = th.scalar_tensor(self.alpha).to(self.device)
        
        # Buffer
        # self.env.observation_space.dtype = np.float32
        
        self.buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
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

    def report_time(self, elapsed_time, stage, file):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        with open(file, 'a') as outfile:
            outfile.write(f'Runtime for {stage}: {hours} hours, {minutes} minutes, {seconds:.2f} seconds\n')
    
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
            # a_lr=self.a_lr,
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
            if isinstance(value, th.Tensor):
                obs[key] = value.to(self.device).unsqueeze(0)
            else:
                obs[key] = value.to(self.device)
        with th.no_grad():
            action, _, _ = self.actor.get_action(obs)

        return action[0].detach().cpu().numpy()

    @override
    def update(self):
        (mb_obs, mb_act, mb_rewards, mb_next_obs, mb_dones) = self.buffer.sample(
            self.batch_size, device=self.device, use_cer=False
        )# sample a batch of experiences from the buffer
        # print('mb_obs',mb_obs['timing_graph'].ndata['feature'].shape)
        
        with th.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(mb_next_obs)
            # (!) Q values are scalarized before being compared (min of ensemble networks)
            # print('mb_next_obs',mb_next_obs['timing_graph'].ndata['feature'].shape)
            # print(self.qf1_target(mb_next_obs).shape)
            
            qf1_next_target = self.scalarization(self.qf1_target(mb_next_obs, next_state_actions), self.weights_tensor)
            qf2_next_target = self.scalarization(self.qf2_target(mb_next_obs, next_state_actions), self.weights_tensor)
            min_qf_next_target = th.min(qf1_next_target, qf2_next_target) - (self.alpha_tensor * next_state_log_pi).flatten()
            scalarized_rewards = self.scalarization(mb_rewards, self.weights_tensor)
            next_q_value = scalarized_rewards.flatten() + (~mb_dones.flatten()).float() * self.gamma * min_qf_next_target

        qf1_a_values = self.scalarization(self.qf1(mb_obs, mb_act), self.weights_tensor).flatten()
        qf2_a_values = self.scalarization(self.qf2(mb_obs, mb_act), self.weights_tensor).flatten()
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        self.q_optimizer.step()
        # GPU_Memory_Monitor()
        
        if self.global_step % self.policy_freq == 0:  # TD 3 Delayed update support
            for _ in range(self.policy_freq):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = self.actor.get_action(mb_obs)
                # (!) Q values are scalarized before being compared (min of ensemble networks)
                qf1_pi = self.scalarization(self.qf1(mb_obs, pi), self.weights_tensor)
                qf2_pi = self.scalarization(self.qf2(mb_obs, pi), self.weights_tensor)
                min_qf_pi = th.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((self.alpha_tensor * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune: # automatic tuning of alpha
                    with th.no_grad():
                        _, log_pi, _ = self.actor.get_action(mb_obs)
                    alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha_tensor = self.log_alpha.exp()
                    self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if self.global_step % self.target_net_freq == 0:
            polyak_update(params=self.qf1.parameters(), target_params=self.qf1_target.parameters(), tau=self.tau)
            polyak_update(params=self.qf2.parameters(), target_params=self.qf2_target.parameters(), tau=self.tau)
            self.qf1_target.requires_grad_(False)
            self.qf2_target.requires_grad_(False)
        
        if self.global_step % 10 == 0 and self.log:
            log_str = f"_{self.id}" if self.id is not None else ""
            to_log = {
                f"losses{log_str}/alpha": self.alpha,
                f"losses{log_str}/qf1_values": qf1_a_values.mean().item(),
                f"losses{log_str}/qf2_values": qf2_a_values.mean().item(),
                f"losses{log_str}/qf1_loss": qf1_loss.item(),
                f"losses{log_str}/qf2_loss": qf2_loss.item(),
                f"losses{log_str}/qf_loss": qf_loss.item() / 2.0,
                f"losses{log_str}/actor_loss": actor_loss.item(),
                "global_step": self.global_step,
            }
            if self.autotune:
                to_log[f"losses{log_str}/alpha_loss"] = alpha_loss.item()
            wandb.log(to_log)
        
    def train(self, total_timesteps: int, eval_env: Optional[gym.Env] = None, start_time=None, reward_log=True):
        """Train the agent.

        Args:
            total_timesteps (int): Total number of timesteps (env steps) to train for
            eval_env (Optional[gym.Env]): Gym environment used for evaluation.
            start_time (Optional[float]): Starting time for the training procedure. If None, it will be set to the current time.
        """
        if start_time is None:
            start_time = time.time()
        end_time = time.time()
        
        csv_filename = f"rewards.csv" 
        if os.path.isfile(csv_filename):
            os.remove(csv_filename) 
        
        runtime_filename = f"runtime" 
        if os.path.isfile(runtime_filename):
            os.remove(runtime_filename) 
        
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
                actions = actions[0].detach().cpu().numpy()

            # execute the game and log data
            next_obs, rewards, terminated = self.env.step(actions)
            print(f'Timing Step: {step}, rewards: {rewards}')

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs
            # if "final_observation" in infos:
            #     real_next_obs = infos["final_observation"]
            self.buffer.add(obs=obs, next_obs=real_next_obs, action=actions, reward=rewards, done=terminated)
            #print(rewards)
            if reward_log == True:
                # Log rewards to a CSV file 
                rewards_to_log = rewards # Convert tensor to numpy array for easier handling 
                file_exists = os.path.isfile(csv_filename) 

                with open(csv_filename, mode='a', newline='') as file: 
                    writer = csv.writer(file) 
                    if not file_exists: 
                        writer.writerow(["obs", "Reward[0]", "Reward[1]"]) # Write header if file doesn't exist 
                    writer.writerow([obs['delta_strength'], rewards_to_log[0], rewards_to_log[1]]) 
                log_str = f"_{self.id}" if self.id is not None else ""
                to_log = {f"rewards{log_str}/Reward[0]": rewards_to_log[0]}
                wandb.log(to_log)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            if terminated:
                obs, _ = self.env.reset()
                # if self.log and "episode" in infos.keys():
                #     log_episode_info(infos["episode"], np.dot, self.weights, self.global_step, self.id)

            # ALGO LOGIC: training.
            if self.global_step > self.learning_starts:
                self.update()
                # print(self.global_step)
                if self.log and self.global_step % 5 == 0:
                    # print('Time consumed:', time.time() - start_time)
                    wandb.log(
                        {"charts/SPS": float(self.global_step / (time.time() - start_time)), "global_step": self.global_step}
                    )
            
            time_for_step = time.time() - end_time
            end_time = time.time()
            runtime = time.time() - start_time
            self.report_time(time_for_step, 'current step', runtime_filename)
            self.report_time(runtime, 'total step', runtime_filename)
            self.global_step += 1
