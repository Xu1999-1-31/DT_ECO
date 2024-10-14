import torch as th
import numpy as np
import dgl
# import copy

# def clone_graph(g):
#     # distinguish graph and heterograph
#     if len(g.etypes) > 1:
#         # 创建一个空的异质图，确保边类型包含源节点类型和目标节点类型
#         g_clone = dgl.heterograph({
#             g.to_canonical_etype(etype): (g.edges(etype=etype)[0], g.edges(etype=etype)[1]) for etype in g.etypes
#         }, num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})
        
#         # 复制每种边类型的边特性
#         for etype in g.etypes:
#             for key, value in g.edges[etype].data.items():
#                 g_clone.edges[etype].data[key] = value.clone()

#         # 复制每种节点类型的节点特性
#         for ntype in g.ntypes:
#             for key, value in g.nodes[ntype].data.items():
#                 g_clone.nodes[ntype].data[key] = value.clone()
    
#     else:
#         g_clone = dgl.graph((g.edges()[0], g.edges()[1]), num_nodes=g.num_nodes())
        
#         for key, value in g.ndata.items():
#             g_clone.ndata[key] = value.clone()
#         for key, value in g.edata.items():
#             g_clone.edata[key] = value.clone()
    
#     return g_clone

class ReplayBuffer:
    """Replay buffer for multi-objective reinforcement learning with structured observation space using PyTorch tensors for observations and gate_features, NumPy for actions, rewards, done, and gate_sizes."""

    def __init__(
        self,
        obs_shape,
        action_dim,
        rew_dim=1,
        max_size=100,
        device='cpu',
    ):
        """Initialize the replay buffer with mixed data types (PyTorch for observations and gate_features, NumPy for actions, rewards, done, and gate_sizes)."""
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        self.device = device
        
        # Initialize buffer for each part of the observation space, the original data are saved in CPU memory
        self.gate_sizes = th.zeros((max_size,) + obs_shape['gate_sizes'], dtype=th.float32, device='cpu')  # gate_sizes as NumPy array
        self.layout = th.zeros((max_size,) + obs_shape['layout'], dtype=th.float32, device='cpu')
        self.padding_mask = th.zeros((max_size,) + obs_shape['padding_mask'], dtype=th.float32, device='cpu')
        self.gate_features = th.zeros((max_size,) + obs_shape['gate_features'], dtype=th.float32, device='cpu')

        # Buffer for storing DGL graph objects (timing graph)
        self.timing_graph = [None] * max_size  # List for DGL graphs

        # Initialize buffer for next observations
        self.next_gate_sizes = th.zeros_like(self.gate_sizes)  # gate_sizes as NumPy array
        self.next_layout = th.zeros_like(self.layout)
        self.next_padding_mask = th.zeros_like(self.padding_mask)
        self.next_gate_features = th.zeros_like(self.gate_features)
        self.next_timing_graph = [None] * max_size  # List for next DGL graphs

        # Initialize buffer for actions, rewards, and done signals using NumPy arrays
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.bool_)  # Boolean array for done signals

    def add(self, obs, action, reward, next_obs, done):
        """Add a new experience to the buffer, using mixed data types."""
        # Store current observation components
        self.gate_sizes[self.ptr] = obs['gate_sizes'].clone().to('cpu')  # Store gate_sizes as NumPy array
        self.layout[self.ptr] = obs['layout'].clone().to('cpu')
        self.padding_mask[self.ptr] = obs['padding_mask'].clone().to('cpu')
        self.gate_features[self.ptr] = obs['gate_features'].clone().to('cpu')
        self.timing_graph[self.ptr] = obs['timing_graph'].to('cpu')  # Store DGL graph object directly

        # Store next observation components
        self.next_gate_sizes[self.ptr] = next_obs['gate_sizes'].clone().to('cpu')  # Next gate_sizes as NumPy array
        self.next_layout[self.ptr] = next_obs['layout'].clone().to('cpu')
        self.next_padding_mask[self.ptr] = next_obs['padding_mask'].clone().to('cpu')
        self.next_gate_features[self.ptr] = next_obs['gate_features'].clone().to('cpu')
        self.next_timing_graph[self.ptr] = next_obs['timing_graph'].to('cpu')  # Store next DGL graph object

        # Store action, reward, and done (NumPy arrays)
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()  # Store done as NumPy boolean array

        # Update buffer pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, device=None):
        """Sample a batch of experiences, batch DGL graphs, and return PyTorch tensors for observations and NumPy arrays for actions, rewards, and done."""
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience for CER

        # Create a dictionary for observations (PyTorch tensors and NumPy arrays)
        observations = {
            'gate_sizes': self.gate_sizes[inds].to(self.device),  # gate_sizes as NumPy array
            'layout': self.layout[inds].to(self.device),
            'padding_mask': self.padding_mask[inds].to(self.device),
            'gate_features': self.gate_features[inds].clone().detach().to(self.device),
        }

        # Create a dictionary for next observations
        next_observations = {
            'gate_sizes': self.next_gate_sizes[inds].to(self.device),  # Next gate_sizes as NumPy array
            'layout': self.next_layout[inds].to(self.device),
            'padding_mask': self.next_padding_mask[inds].to(self.device),
            'gate_features': self.next_gate_features[inds].clone().detach().to(self.device),
        }

        # Batch the DGL graphs for both current and next observations
        batched_graph = dgl.batch([self.timing_graph[i] for i in inds]).to(self.device)
        next_batched_graph = dgl.batch([self.next_timing_graph[i] for i in inds]).to(self.device)
        
        observations['timing_graph'] = batched_graph
        next_observations['timing_graph'] = next_batched_graph

        # Convert actions, rewards, and done from NumPy to PyTorch tensors, if needed
        actions = th.tensor(self.actions[inds], dtype=th.float32, device=device)
        rewards = th.tensor(self.rewards[inds], dtype=th.float32, device=device)
        dones = th.tensor(self.dones[inds], dtype=th.bool, device=device)

        experience_tuples = (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )

        return experience_tuples

    def __len__(self):
        """Get the size of the buffer."""
        return self.size

