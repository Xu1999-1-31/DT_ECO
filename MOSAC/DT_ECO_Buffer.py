import numpy as np
import torch as th

## Buffer For DTECO Observation Space
class ReplayBuffer:
    """Replay buffer for multi-objective reinforcement learning with structured observation space."""

    def __init__(
        self,
        obs_shape,
        action_dim,
        rew_dim=1,
        max_size=100000,
        obs_dtype=np.float32,
        action_dtype=np.float32,
    ):
        """Initialize the replay buffer.

        Args:
            obs_shape: Dictionary containing the shape of the observation space for "gate_sizes", "timing_graph", "layout", and "padding_mask"
            action_dim: Dimension of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            obs_dtype: Data type of the observations
            action_dtype: Data type of the actions
        """
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        
        # Initialize buffer for each part of the observation space
        self.gate_sizes = np.zeros((max_size,) + obs_shape["gate_sizes"], dtype=np.int32)
        self.timing_graph = np.zeros((max_size,) + obs_shape["timing_graph"], dtype=obs_dtype)
        self.layout = np.zeros((max_size,) + obs_shape["layout"], dtype=obs_dtype)
        self.padding_mask = np.zeros((max_size,) + obs_shape["padding_mask"], dtype=obs_dtype)

        # Initialize buffer for next observations
        self.next_gate_sizes = np.zeros_like(self.gate_sizes)
        self.next_timing_graph = np.zeros_like(self.timing_graph)
        self.next_layout = np.zeros_like(self.layout)
        self.next_padding_mask = np.zeros_like(self.padding_mask)

        # Initialize buffer for actions, rewards, and done signals
        self.actions = np.zeros((max_size, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        """Add a new experience to the buffer.

        Args:
            obs: Observation (with "gate_sizes", "timing_graph", "layout", and "padding_mask")
            action: Action
            reward: Reward
            next_obs: Next observation (with "gate_sizes", "timing_graph", "layout", and "padding_mask")
            done: Done
        """
        # Store current observation components
        self.gate_sizes[self.ptr] = np.array(obs["gate_sizes"]).copy()
        self.timing_graph[self.ptr] = np.array(obs["timing_graph"]).copy()
        self.layout[self.ptr] = np.array(obs["layout"]).copy()
        self.padding_mask[self.ptr] = np.array(obs["padding_mask"]).copy()

        # Store action, reward, and done signals
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()

        # Store next observation components
        self.next_gate_sizes[self.ptr] = np.array(next_obs["gate_sizes"]).copy()
        self.next_timing_graph[self.ptr] = np.array(next_obs["timing_graph"]).copy()
        self.next_layout[self.ptr] = np.array(next_obs["layout"]).copy()
        self.next_padding_mask[self.ptr] = np.array(next_obs["padding_mask"]).copy()

        # Update buffer pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None):
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: Batch size
            replace: Whether to sample with replacement
            use_cer: Whether to use CER (Critic Experience Replay)
            to_tensor: Whether to convert the data to PyTorch tensors
            device: Device to use

        Returns:
            A tuple of (observations, actions, rewards, next observations, dones)
        """
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience for CER

        # Create a dictionary for observations
        observations = {
            "gate_sizes": self.gate_sizes[inds],
            "timing_graph": self.timing_graph[inds],
            "layout": self.layout[inds],
            "padding_mask": self.padding_mask[inds],
        }
        
        # Create a dictionary for next observations
        next_observations = {
            "gate_sizes": self.next_gate_sizes[inds],
            "timing_graph": self.next_timing_graph[inds],
            "layout": self.next_layout[inds],
            "padding_mask": self.next_padding_mask[inds],
        }

        experience_tuples = (
            observations,
            self.actions[inds],
            self.rewards[inds],
            next_observations,
            self.dones[inds],
        )

        if to_tensor:
            # Convert the experience tuples to PyTorch tensors if required
            return tuple(map(lambda x: {k: th.tensor(v, device=device) for k, v in x.items()} if isinstance(x, dict) else th.tensor(x, device=device), experience_tuples))
        else:
            return experience_tuples

    def __len__(self):
        """Get the size of the buffer."""
        return self.size