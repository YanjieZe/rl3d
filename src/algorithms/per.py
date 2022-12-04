# PER: Priorited Replay Buffer

import torch
import numpy as np
import random
import utils
from termcolor import colored



def prefill(arr, chunk_size=20_000):
	capacity = arr.shape[0]
	for i in range(0, capacity+1, chunk_size):
		chunk = min(chunk_size, capacity-i)
		arr[i:i+chunk] = np.random.randn(chunk, *arr.shape[1:])
		
	return arr

def prefill_memory(capacity, obs_shape):
	obses = []
	if len(obs_shape) > 1:
		c, h, w = obs_shape
		for _ in range(capacity):
			frame = np.ones((c, h, w), dtype=np.uint8)
			obses.append(frame)
	else:
		for _ in range(capacity):
			obses.append(np.ones(obs_shape[0], dtype=np.float32))
	print(colored("prefill replay buffer...", color="cyan") )
	return np.stack(obses, axis=0)


class EfficientPrioritizedReplayBuffer():
	def __init__(self,
		obs_shape: tuple,
		state_shape: tuple,
		action_shape: tuple,
		capacity: int,
		batch_size: int,
		prioritized_replay: bool,
		alpha: float,
		beta: float,
		ensemble_size: int,
		device: torch.device='cuda',
		prefilled=True,
		episode_length=50,
		observation_type="image",
		use_single_image=False,
		):
		self.capacity = capacity
		self.batch_size = batch_size
		self.prioritized_replay = prioritized_replay
		self.device = device
		self.state_shape = state_shape
		self.episode_length = episode_length
		self.obs_shape = obs_shape
		self.observation_type = observation_type

		state = len(obs_shape) == 1

		self.use_single_image = use_single_image
		if self.use_single_image:
			obs_shape = list(obs_shape)
			obs_shape[0] = int(obs_shape[0] / 2)
			obs_shape = tuple(obs_shape)

		if prefilled:
			self._obs = prefill_memory(capacity, obs_shape)
			self._last_obs = prefill_memory(capacity//self.episode_length, obs_shape)
			if self.state_shape:
				self._state = prefill_memory(capacity, state_shape)
				self._last_state = prefill_memory(capacity//self.episode_length, state_shape)

		else:
			self._obs = np.empty((capacity, *obs_shape), dtype=np.float32 if state else np.uint8)
			self._last_obs = np.empty((capacity, *obs_shape), dtype=np.float32 if state else np.uint8)

			if self.state_shape:
				self._state = np.empty((capacity, *state_shape), dtype=np.float32)
				self._last_state = np.empty((capacity, *state_shape), dtype=np.float32)
	
		self._action = prefill(np.empty((capacity, *action_shape), dtype=np.float32))
		self._reward = prefill(np.empty((capacity,), dtype=np.float32))
	
		self._alpha = alpha
		self._beta = beta
		self._ensemble_size = ensemble_size
		self._priority_eps = 1e-6
		self._priorities = np.ones((capacity, ensemble_size), dtype=np.float32)

		self.ep_idx = 0
		self.idx = 0
		self.full = False

	def obs_process(self, obs):
		if self.observation_type in ["image", "state+image"]:
			return torch.as_tensor(obs).cuda().float().div(255)
		elif self.observation_type == "state":
			return torch.as_tensor(obs).cuda().float()
		else:
			raise NotImplementedError
			
	def add(self, obs, state, action, reward, next_obs, next_state):
		obs = obs[:3] if self.use_single_image else obs
		next_obs = next_obs[:3] if self.use_single_image else next_obs
		self._obs[self.idx] = obs
		self._action[self.idx] = action
		self._reward[self.idx] = reward

		if (self.idx+1)%self.episode_length==0:
			self._last_obs[self.idx//self.episode_length] = next_obs 
			if self.state_shape:
				self._last_state[self.idx//self.episode_length] = next_state

		if self.state_shape:
			self._state[self.idx] = state

		if self.prioritized_replay:
			if self.full:
				self._max_priority = self._priorities.max()
			elif self.idx == 0:
				self._max_priority = 1.0
			else:
				self._max_priority = self._priorities[:self.idx].max()
			new_priorities = self._max_priority # the latest one has the max priority
			self._priorities[self.idx] = new_priorities

		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	@property
	def max_priority(self):
		return self._max_priority if self.prioritized_replay else 0

	def update_priorities(self, idxs, priorities:np.ndarray, idx:int=None):
		if not self.prioritized_replay:
			return
		self._priorities[idxs, idx] = priorities + self._priority_eps # add epsilon for > 0

	def get_next_obs(self, idxs):

		new_idxs = (idxs+1) % self.capacity
		next_obs = []
		for idx in new_idxs:
			if ( (idx) % self.episode_length==0):
				next_obs.append( self._last_obs[ (idx)//self.episode_length -1])
			else:
				next_obs.append(self._obs[idx])
		return np.stack(next_obs, axis=0)


	def get_next_state(self, idxs):
		new_idxs = (idxs+1) % self.capacity
		next_state = []
		for idx in new_idxs:
			if ( (idx) % self.episode_length==0):
				next_state.append( self._last_state[(idx)//self.episode_length -1])
			else:
				next_state.append(self._state[idx])
		return np.stack(next_state, axis=0)


    		
	def uniform_sample(self, idx=None):
		if idx is None:
			idx = 0
		# uniform sampling
		probs = self._priorities[:, idx] if self.full else self._priorities[:self.idx, idx]
		probs[:] = 1
		probs[self.idx-1] = 0
		probs /= probs.sum()
		total = len(probs)
		idxs = np.random.choice(total, self.batch_size, p=probs, replace=False)
		obs, next_obs = self._obs[idxs], self.get_next_obs(idxs)

		obs = self.obs_process(obs)
		next_obs = self.obs_process(next_obs)
		actions = torch.as_tensor(self._action[idxs]).cuda()
		rewards = torch.as_tensor(self._reward[idxs]).cuda()

		if self.state_shape:
			state, next_state = self._state[idxs], self.get_next_state(idxs)
			state, next_state = torch.as_tensor(state).cuda(), torch.as_tensor(next_state).cuda()
		else:
			state, next_state = None, None

		return obs, state, actions, rewards.unsqueeze(1), next_obs, next_state

	def prioritized_sample(self, idx=None):
		if idx is None:
			idx = 0
		probs =self._priorities[:, idx]** self._alpha if self.full else self._priorities[:self.idx, idx] ** self._alpha
		probs[self.idx-1] = 0
		probs /= probs.sum()
		total = len(probs)
		idxs = np.random.choice(total, self.batch_size, p=probs, replace=False)
		weights = (total * probs[idxs]) ** (-self._beta)
		weights /= weights.max()

		obs, next_obs = self._obs[idxs], self.get_next_obs(idxs)
		obs = self.obs_process(obs)
		next_obs = self.obs_process(next_obs)


		actions = torch.as_tensor(self._action[idxs]).cuda()
		rewards = torch.as_tensor(self._reward[idxs]).cuda()

		if self.state_shape:
			state, next_state = self._state[idxs], self.get_next_state(idxs)
			state, next_state = torch.as_tensor(state).cuda(), torch.as_tensor(next_state).cuda()
		else:
			state, next_state = None, None

		return obs, state, actions, rewards.unsqueeze(1), next_obs, next_state, idxs, weights

	def sample(self, idx=None, step=None):
		if self.prioritized_replay:
			return self.prioritized_sample(idx)
		else:
			return self.uniform_sample(idx)

	def save(self, fp=None):
		pass
