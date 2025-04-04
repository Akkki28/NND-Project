import numpy as np
import matplotlib.pyplot as plt
import random
import simpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import collections
from collections import defaultdict
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
import os
import math
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import time

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

SLICES = [
    {"name": "MIoT", "bandwidth": 95, "latency": 13},
    {"name": "eMBB", "bandwidth": 15, "latency": 9},
    {"name": "HMTC", "bandwidth": 1.2, "latency": 7},
    {"name": "URLLC", "bandwidth": 1.3, "latency": 4},
    {"name": "V2X", "bandwidth": 18, "latency": 11}
]

TRAFFIC_PROFILES = [
    {"name": "Video", "load_range": (0.8, 2.0), "preferred_slice": 1},  
    {"name": "Audio", "load_range": (0.2, 0.8), "preferred_slice": 4},  
    {"name": "IoT", "load_range": (0.1, 0.3), "preferred_slice": 0},    
    {"name": "WebData", "load_range": (0.5, 1.5), "preferred_slice": 2}, 
    {"name": "ControlData", "load_range": (0.3, 0.7), "preferred_slice": 3}, 
    {"name": "BestEffort", "load_range": (0.1, 1.0), "preferred_slice": 2}  
]

class NetworkSlicingEnv(gym.Env):
    def __init__(self, max_steps=10000, arrival_rate=0.3):
        super(NetworkSlicingEnv, self).__init__()
        
        self.max_steps = max_steps
        self.arrival_rate = arrival_rate
        self.current_step = 0
        
        self.slices = SLICES.copy()
        self.traffic_profiles = TRAFFIC_PROFILES.copy()
        self.num_slices = len(self.slices)
        self.num_profiles = len(self.traffic_profiles)
        
        self.action_space = spaces.Discrete(self.num_slices * self.num_slices)
        
        obs_size = self.num_slices + (self.num_slices * self.num_profiles) + self.num_slices
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=(obs_size,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.current_step = 0
        self.ues = []
        self.slice_loads = np.zeros(self.num_slices)
        self.ue_count_per_slice = np.zeros(self.num_slices, dtype=int)
        self.ue_types_per_slice = np.zeros((self.num_slices, self.num_profiles), dtype=int)
        
        self.rejected_ues = 0
        self.total_ues = 0
        self.qos_violations = 0
        
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        self.current_step += 1
        
        source_slice = action // self.num_slices
        target_slice = action % self.num_slices
        
        self._process_arrivals()
        
        reward = self._process_movement(source_slice, target_slice)
        
        self._update_network_state()
        
        done = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        
        info = {
            "rejected_ratio": self.rejected_ues / max(1, self.total_ues),
            "qos_violations": self.qos_violations,
            "slice_loads": self.slice_loads.copy(),
            "ues": self.ues.copy(),
            "ue_count_per_slice": self.ue_count_per_slice.copy()
        }
        
        return observation, reward, done, False, info
    
    def _process_arrivals(self):
        num_arrivals = np.random.poisson(self.arrival_rate)

        if self.current_step % 10 == 0: 
            for profile_idx in range(self.num_profiles):
                profile = self.traffic_profiles[profile_idx]
                min_load, max_load = profile["load_range"]
                load = random.uniform(min_load, max_load)
                self._add_specific_ue(profile_idx, load)
        
        for _ in range(num_arrivals):
            self._add_ue()
    
    def _add_ue(self):
        profile_idx = random.randint(0, self.num_profiles - 1)
        profile = self.traffic_profiles[profile_idx]
        
        min_load, max_load = profile["load_range"]
        load = random.uniform(min_load, max_load)
        preferred_slice = profile["preferred_slice"]
        
        ue = {
            "profile": profile_idx,
            "load": load,
            "preferred_slice": preferred_slice,
            "allocated_slice": None,
            "time_in_network": 0,
            "id": self.total_ues
        }
        
        self.total_ues += 1
        
        if self.slice_loads[preferred_slice] + load <= self.slices[preferred_slice]["bandwidth"]:
            allocated_slice = preferred_slice
        else:
            allocated_slice = -1
            for slice_idx in range(self.num_slices):
                if slice_idx != preferred_slice and self.slice_loads[slice_idx] + load <= self.slices[slice_idx]["bandwidth"]:
                    allocated_slice = slice_idx
                    break
        
        if allocated_slice >= 0:
            ue["allocated_slice"] = allocated_slice
            self.ues.append(ue)
            self.slice_loads[allocated_slice] += load
            self.ue_count_per_slice[allocated_slice] += 1
            self.ue_types_per_slice[allocated_slice, profile_idx] += 1
        else:
            self.rejected_ues += 1

    
    def _add_specific_ue(self, profile_idx, load):
        profile = self.traffic_profiles[profile_idx]
        preferred_slice = profile["preferred_slice"]
        
        ue = {
            "profile": profile_idx,
            "load": load,
            "preferred_slice": preferred_slice,
            "allocated_slice": None,
            "time_in_network": 0,
            "id": self.total_ues
        }
        
        self.total_ues += 1
        
        slice_options = [(preferred_slice, 0)] 
        
        for slice_idx in range(self.num_slices):
            if slice_idx != preferred_slice:
                penalty = 0.1  
                slice_options.append((slice_idx, penalty))
        
        slice_options.sort(key=lambda x: (self.slice_loads[x[0]] / self.slices[x[0]]["bandwidth"]) + x[1])
        
        for slice_idx, _ in slice_options:
            if self.slice_loads[slice_idx] + load <= self.slices[slice_idx]["bandwidth"]:
                ue["allocated_slice"] = slice_idx
                self.ues.append(ue)
                self.slice_loads[slice_idx] += load
                self.ue_count_per_slice[slice_idx] += 1
                self.ue_types_per_slice[slice_idx, profile_idx] += 1
                return
            
        self.rejected_ues += 1

        slice_options = [(preferred_slice, 0)] 
    
        for slice_idx in range(self.num_slices):
            if slice_idx != preferred_slice:
                penalty = 0.1 
                slice_options.append((slice_idx, penalty))
        
        slice_options.sort(key=lambda x: (self.slice_loads[x[0]] / self.slices[x[0]]["bandwidth"]) + x[1])
        
        for slice_idx, _ in slice_options:
            if self.slice_loads[slice_idx] + load <= self.slices[slice_idx]["bandwidth"]:
                ue["allocated_slice"] = slice_idx
                self.ues.append(ue)
                self.slice_loads[slice_idx] += load
                self.ue_count_per_slice[slice_idx] += 1
                self.ue_types_per_slice[slice_idx, profile_idx] += 1
                return
        
        self.rejected_ues += 1
    
    def _process_movement(self, source_slice, target_slice):
        reward = 0
        
        if source_slice == target_slice:
            return -1
        
        if self.ue_count_per_slice[source_slice] == 0:
            return -1
        
        ues_to_move = []
        moved_load = 0
        
        for ue in self.ues:
            if ue["allocated_slice"] == source_slice:
                ues_to_move.append(ue)
                moved_load += ue["load"]
        
        if self.slice_loads[target_slice] + moved_load <= self.slices[target_slice]["bandwidth"]:
            before_utilization = np.array([
                self.slice_loads[i] / self.slices[i]["bandwidth"] 
                for i in range(self.num_slices)
            ])
            
            for ue in ues_to_move:
                self.ue_types_per_slice[source_slice, ue["profile"]] -= 1
                self.ue_types_per_slice[target_slice, ue["profile"]] += 1
                ue["allocated_slice"] = target_slice
            
            self.slice_loads[source_slice] -= moved_load
            self.slice_loads[target_slice] += moved_load
            
            self.ue_count_per_slice[source_slice] -= len(ues_to_move)
            self.ue_count_per_slice[target_slice] += len(ues_to_move)
            after_utilization = np.array([
                self.slice_loads[i] / self.slices[i]["bandwidth"] 
                for i in range(self.num_slices)
            ])
            
            before_std = np.std(before_utilization)
            after_std = np.std(after_utilization)
            balance_reward = 0.5 * (before_std - after_std)
            
            correct_slice_moves = sum(1 for ue in ues_to_move if ue["preferred_slice"] == target_slice)
            incorrect_slice_moves = len(ues_to_move) - correct_slice_moves
            preference_reward = 0.5 * (correct_slice_moves - incorrect_slice_moves*0.5)
            low_utilization_slices = after_utilization < 0.3  
            utilization_reward = 0
            if low_utilization_slices[target_slice]:
                utilization_reward = 1.5 
            reward = balance_reward + preference_reward + utilization_reward
            
            if before_utilization[source_slice] > 0.9:
                reward += 2
        else:
            reward = -2
        
        return reward
    
    def _update_network_state(self):
        for ue in self.ues:
            ue["time_in_network"] += 1
        
        ues_to_remove = []
        for i, ue in enumerate(self.ues):
            leave_prob = min(0.05, 0.001 * ue["time_in_network"])
            if random.random() < leave_prob:
                ues_to_remove.append(i)
        
        for i in sorted(ues_to_remove, reverse=True):
            ue = self.ues[i]
            slice_idx = ue["allocated_slice"]
            
            self.slice_loads[slice_idx] -= ue["load"]
            self.ue_count_per_slice[slice_idx] -= 1
            self.ue_types_per_slice[slice_idx, ue["profile"]] -= 1
            
            self.ues.pop(i)
        
        for i, slice_info in enumerate(self.slices):
            if self.slice_loads[i] > slice_info["bandwidth"]:
                overload_ratio = self.slice_loads[i] / slice_info["bandwidth"] - 1
                self.qos_violations += int(overload_ratio * 10) * self.ue_count_per_slice[i]
    
    def _get_observation(self):
        normalized_loads = np.array([
            self.slice_loads[i] / self.slices[i]["bandwidth"] 
            for i in range(self.num_slices)
        ])
        
        ue_types_flat = self.ue_types_per_slice.flatten() / max(1, np.max(self.ue_types_per_slice))
        
        observation = np.concatenate([
            self.ue_count_per_slice / max(1, np.max(self.ue_count_per_slice)),
            ue_types_flat,
            normalized_loads
        ])
        
        return observation.astype(np.float32)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, x):
        shared = self.shared(x)
        policy = self.policy_head(shared)
        value = self.value_head(shared)
        return policy, value

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, 
                 batch_size=64, target_update=10, device="cpu"):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        
        self.q_network = DQNNetwork(state_dim, action_dim).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.buffer = ReplayBuffer(buffer_size)
        
        self.train_step = 0
    
    def select_action(self, state, evaluation=False):
        if not evaluation and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
    
    def train(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = F.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, 
                 alpha=0.2, buffer_size=10000, batch_size=64, device="cpu"):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
        
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(device)
        self.target_critic1 = CriticNetwork(state_dim, action_dim).to(device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim).to(device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state, evaluation=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if evaluation:
                action_probs = self.actor(state_tensor)
                return action_probs.argmax(1).item()
            else:
                action_probs = self.actor(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample()
                return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
    
    def train(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_q1 = self.target_critic1(next_states)
            next_q2 = self.target_critic2(next_states)
            next_q = torch.min(next_q1, next_q2)
            expected_q = torch.sum(next_action_probs * (next_q - self.alpha * torch.log(next_action_probs + 1e-8)), dim=1)
            target_q = rewards + (1 - dones) * self.gamma * expected_q
        
        current_q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss_q1 = F.mse_loss(current_q1, target_q)
        self.critic1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic1_optimizer.step()
        
        current_q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss_q2 = F.mse_loss(current_q2, target_q)
        self.critic2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic2_optimizer.step()
        
        action_probs = self.actor(states)
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        q = torch.min(q1, q2)
        
        actor_loss = -torch.mean(torch.sum(action_probs * (q - self.alpha * torch.log(action_probs + 1e-8)), dim=1))
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return (loss_q1.item() + loss_q2.item() + actor_loss.item()) / 3.0

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, 
                 epochs=10, batch_size=64, device="cpu"):
        self.policy = PPONetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def select_action(self, state, evaluation=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.policy(state_tensor)
            
            if evaluation:
                return action_probs.argmax(1).item()
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                self.states.append(state)
                self.actions.append(action.item())
                self.log_probs.append(log_prob.item())
                self.values.append(value.item())
                
                return action.item()
    
    def store_transition(self, reward, next_state, done):
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def train(self):
        if len(self.states) == 0:
            return 0.0
        
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        returns = []
        advantages = []
        
        next_state = self.next_states[-1]
        if not self.dones[-1]:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, next_value = self.policy(next_state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0
        
        gae = 0
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * 0.95 * (1 - self.dones[i]) * gae
            returns.insert(0, gae + self.values[i])
            advantages.insert(0, gae)
            next_value = self.values[i]
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                action_probs, values = self.policy(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                
                surrogate1 = ratio * batch_advantages
                surrogate2 = clipped_ratio * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        return total_loss / (len(states) / self.batch_size * self.epochs)

class HeuristicAllocator:
    def __init__(self, env):
        self.env = env
        self.num_slices = len(env.slices)
    
    def select_action(self, observation, evaluation=False):
        slice_loads = observation[-self.num_slices:]
        
        source_slice = np.argmax(slice_loads)
        target_slice = np.argmin(slice_loads)
        
        action = source_slice * self.num_slices + target_slice
        
        return action

class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
    
    def select_action(self, state, evaluation=False):
        return random.randint(0, self.action_dim - 1)

def train_agent(agent_name, env, num_episodes=100, eval_interval=10):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if agent_name == "dqn":
        agent = DQNAgent(obs_dim, action_dim, device=device)
    elif agent_name == "sac":
        agent = SACAgent(obs_dim, action_dim, device=device)
    elif agent_name == "ppo":
        agent = PPOAgent(obs_dim, action_dim, device=device)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    episode_rewards = []
    eval_rewards = []
    training_losses = []
    
    for episode in tqdm(range(1, num_episodes + 1), desc=f"Training {agent_name}"):
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        while not done:
            action = agent.select_action(observation)
            
            next_observation, reward, done, _, info = env.step(action)
            
            if agent_name == "ppo":
                agent.store_transition(reward, next_observation, done)
            else:
                agent.store_transition(observation, action, reward, next_observation, done)
            
            observation = next_observation
            
            if agent_name != "ppo" or done:
                loss = agent.train()
                episode_loss += loss
            
            episode_reward += reward
            steps += 1
        
        if hasattr(agent, 'update_epsilon'):
            agent.update_epsilon()
        
        episode_rewards.append(episode_reward)
        training_losses.append(episode_loss / max(1, steps))
        
        if episode % eval_interval == 0:
            eval_reward, _, _ = evaluate_agent(agent, env, num_episodes=5)
            eval_rewards.append(eval_reward)
            
            print(f"Episode {episode}: Avg Reward: {eval_reward:.2f}")
    
    if agent_name == "sac":
        torch.save(agent.actor.state_dict(), f"models/{agent_name}_final.pth")
    elif hasattr(agent, 'policy'):
        torch.save(agent.policy.state_dict(), f"models/{agent_name}_final.pth")
    else:
        torch.save(agent.q_network.state_dict(), f"models/{agent_name}_final.pth")
    
    np.save(f"results/{agent_name}_episode_rewards.npy", np.array(episode_rewards))
    np.save(f"results/{agent_name}_eval_rewards.npy", np.array(eval_rewards))
    np.save(f"results/{agent_name}_training_losses.npy", np.array(training_losses))
    
    return agent, {
        "episode_rewards": episode_rewards,
        "eval_rewards": eval_rewards,
        "training_losses": training_losses
    }

def evaluate_agent(agent, env, num_episodes=10):
    total_rewards = 0
    total_rejections = 0
    total_qos_violations = 0
    
    for _ in range(num_episodes):
        observation, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(observation, evaluation=True)
            
            observation, reward, done, _, info = env.step(action)
            episode_reward += reward
        
        total_rewards += episode_reward
        total_rejections += info["rejected_ratio"]
        total_qos_violations += info["qos_violations"]
    
    avg_reward = total_rewards / num_episodes
    avg_rejection = total_rejections / num_episodes
    avg_qos_violations = total_qos_violations / num_episodes
    
    return avg_reward, avg_rejection, avg_qos_violations

def plot_training_curves(results_dict):
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for agent_name, results in results_dict.items():
        plt.plot(results["episode_rewards"], label=agent_name)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for agent_name, results in results_dict.items():
        plt.plot(np.arange(0, len(results["eval_rewards"]) * 10, 10), 
                results["eval_rewards"], label=agent_name)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Evaluation Rewards")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/training_curves_comparison.png")
    plt.show()

def run_real_time_simulation(agent, env, update_interval=100):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    slice_names = [s["name"] for s in SLICES]
    slice_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    traffic_profile_names = [p["name"] for p in TRAFFIC_PROFILES]
    
    steps_history = []
    slice_load_history = [[] for _ in range(len(SLICES))]
    ue_count_history = [[] for _ in range(len(SLICES))]
    
    x = np.arange(len(slice_names))
    bars = ax1.bar(x, np.zeros(len(slice_names)), color=slice_colors, alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(slice_names)
    ax1.set_ylabel('Load (Mbps)')
    ax1.set_title('Current Slice Load')
    
    capacity_lines = []
    for i, slice_info in enumerate(SLICES):
        line = ax1.axhline(y=slice_info["bandwidth"], xmin=i/len(SLICES), xmax=(i+1)/len(SLICES), 
                          color='r', linestyle='--', alpha=0.5)
        capacity_lines.append(line)
    
    lines = []
    for i in range(len(SLICES)):
        line, = ax2.plot([], [], label=slice_names[i], color=slice_colors[i])
        lines.append(line)
    
    max_history_points = 100  
    ax2.set_xlim(0, max_history_points)
    ax2.set_ylim(0, 100) 
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Load (Mbps)')
    ax2.set_title('Slice Load History')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    ax3.set_xlim(-1, len(SLICES))
    ax3.set_ylim(-1, 10)  
    ax3.set_xlabel('Slice')
    ax3.set_title('UE Allocation')
    ax3.set_xticks(range(len(SLICES)))
    ax3.set_xticklabels(slice_names)
    ax3.grid(True)
    
    scatter = ax3.scatter([], [], s=100, c=[], cmap='tab10', vmin=0, vmax=len(TRAFFIC_PROFILES)-1)
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_ticks(np.arange(len(TRAFFIC_PROFILES)) + 0.5)
    cbar.set_ticklabels(traffic_profile_names)
    cbar.set_label('Traffic Profile')
    
    metrics_text = ax4.text(0.5, 0.5, '', ha='center', va='center', fontsize=12, transform=ax4.transAxes)
    
    observation, _ = env.reset(seed=42)  
    frame = 0
    
    def update(frame_num):
        nonlocal frame, observation
        action = agent.select_action(observation, evaluation=True)
        new_observation, reward, done, _, info = env.step(action)
        observation = new_observation
        
        if done:
            observation, _ = env.reset(seed=None)  
        
        steps_history.append(frame)
        for i, load in enumerate(info["slice_loads"]):
            slice_load_history[i].append(load)
        
        for i, count in enumerate(info["ue_count_per_slice"]):
            ue_count_history[i].append(count)
        
        for i, bar in enumerate(bars):
            load = info["slice_loads"][i]
            bar.set_height(load)
            
            capacity = SLICES[i]["bandwidth"]
            if load > capacity:
                bar.set_color('red')
            elif load > capacity * 0.8:
                bar.set_color('orange')
            else:
                bar.set_color(slice_colors[i])
        
        start_idx = max(0, len(steps_history) - max_history_points)
        visible_steps = steps_history[start_idx:]
        ax2.set_xlim(max(0, frame - max_history_points), max(max_history_points, frame))
        
        for i, line in enumerate(lines):
            visible_loads = slice_load_history[i][start_idx:]
            line.set_data(visible_steps, visible_loads)
        
        visible_loads = [history[start_idx:] for history in slice_load_history]
        max_load = max([max(loads, default=0) for loads in visible_loads]) * 1.1
        ax2.set_ylim(0, max(100, max_load))
        
        ue_x = [] 
        ue_y = [] 
        ue_colors = []  
        

        ue_positions = {i: 0 for i in range(len(SLICES))}
        
        for ue in info["ues"]:
            slice_idx = ue["allocated_slice"]
            position = ue_positions[slice_idx]
            ue_positions[slice_idx] += 1
            
            ue_x.append(slice_idx)
            ue_y.append(position)
            ue_colors.append(ue["profile"])
        
        scatter.set_offsets(np.column_stack([ue_x, ue_y]))
        scatter.set_array(np.array(ue_colors))
        
        max_ue_per_slice = max(ue_positions.values()) if ue_positions else 1
        ax3.set_ylim(-1, max(10, max_ue_per_slice + 1))
        
        metrics_text.set_text(f"Step: {frame}\n"
                             f"Total UEs: {env.total_ues}\n"
                             f"Active UEs: {len(info['ues'])}\n")
        
        frame += 1
        
        artists = []
        artists.extend(bars)      
        artists.extend(lines)
        artists.append(scatter)
        artists.append(metrics_text)
        artists.extend(capacity_lines)
        
        return tuple(artists) 
    
    ani = animation.FuncAnimation(fig, update, interval=update_interval, blit=True, cache_frame_data=False)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def run_simpy_simulation(agent, env, sim_time=50, update_interval=1):
    import math
    sim_env = simpy.Environment()

    fig, (ax_bar, ax_net) = plt.subplots(1, 2, figsize=(14, 6))
    
    num_nodes = len(SLICES)
    center = (0, 0)
    radius = 4  
    node_positions = {}
    for i in range(num_nodes):
        angle = 2 * math.pi * i / num_nodes
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        node_positions[i] = (x, y)
    
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edges.append((node_positions[i], node_positions[j]))
    
    def simpy_process():
        observation, _ = env.reset(seed=42)
        while True:
            action = agent.select_action(observation, evaluation=True)
            observation, reward, done, _, info = env.step(action)
            
            ax_bar.clear()
            x_values = list(range(len(info["slice_loads"])))
            bars = ax_bar.bar(x_values, info["slice_loads"], color='skyblue')
            ax_bar.set_xticks(x_values)
            ax_bar.set_xticklabels([s["name"] for s in SLICES])
            ax_bar.set_ylabel("Load (Mbps)")
            ax_bar.set_title(f"Slice Loads - Step: {env.current_step}")
            
            ax_net.clear()
            for (p1, p2) in edges:
                ax_net.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linestyle='--', linewidth=1)
            
            node_radius = 0.6
            for i, slice_info in enumerate(SLICES):
                x, y = node_positions[i]
                load_norm = min(info["slice_loads"][i] / slice_info["bandwidth"], 1.0)
                node_color = plt.cm.Reds(load_norm)
                circle = plt.Circle((x, y), node_radius, color=node_color, ec='black', zorder=2)
                ax_net.add_patch(circle)
                ax_net.text(x, y+node_radius+0.2, slice_info["name"], ha='center', va='bottom', fontsize=10, zorder=3)
            
            ue_x = []
            ue_y = []
            ue_colors = []
            for ue in info["ues"]:
                allocated_slice = ue.get("allocated_slice", -1)
                if allocated_slice is None or allocated_slice < 0:
                    continue
                center_x, center_y = node_positions[allocated_slice]
                r = node_radius * math.sqrt(random.uniform(0, 1))
                theta = random.uniform(0, 2*math.pi)
                pos_x = center_x + r * math.cos(theta)
                pos_y = center_y + r * math.sin(theta)
                ue_x.append(pos_x)
                ue_y.append(pos_y)
                ue_colors.append(ue["profile"])
            
            sc = ax_net.scatter(ue_x, ue_y, c=ue_colors, cmap='tab10', s=100,
                                vmin=0, vmax=len(TRAFFIC_PROFILES)-1, edgecolors='k', zorder=3)
            
            ax_net.set_title("Enhanced Network Diagram: UE Allocation")
            ax_net.set_xlim(-radius-2, radius+2)
            ax_net.set_ylim(-radius-2, radius+2)
            ax_net.set_aspect('equal')
            ax_net.axis('off')
            
            plt.pause(0.1)
            
            if done:
                observation, _ = env.reset()
            
            yield sim_env.timeout(update_interval)
    
    sim_env.process(simpy_process())
    sim_env.run(until=sim_time)


def run_combined_simulation(agent, env, update_interval=100, sim_time=50):
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    ax_rt1 = fig.add_subplot(gs[0, 0])  
    ax_rt2 = fig.add_subplot(gs[0, 1]) 
    ax_rt3 = fig.add_subplot(gs[0, 2]) 

    ax_sim1 = fig.add_subplot(gs[1, :2])  
    ax_sim2 = fig.add_subplot(gs[1, 2])   
    ax_sim2.axis('off')
    
    slice_names = [s["name"] for s in SLICES]
    slice_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    traffic_profile_names = [p["name"] for p in TRAFFIC_PROFILES]
    steps_history = []
    slice_load_history = [[] for _ in range(len(SLICES))]
    
    scatter_rt = ax_rt3.scatter([], [], s=100, c=[], cmap='tab10',
                                  vmin=0, vmax=len(TRAFFIC_PROFILES)-1)
    cbar = plt.colorbar(scatter_rt, ax=ax_rt3)
    cbar.set_ticks(np.arange(len(TRAFFIC_PROFILES)) + 0.5)
    cbar.set_ticklabels(traffic_profile_names)
    cbar.set_label('Traffic Profile')
    
    num_nodes = len(SLICES)
    center = (0, 0)
    radius = 4
    node_positions = {}
    for i in range(num_nodes):
        angle = 2 * math.pi * i / num_nodes
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        node_positions[i] = (x, y)

    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edges.append((node_positions[i], node_positions[j]))
    
    observation, _ = env.reset(seed=42)
    frame = 0
    
    def update(frame_num):
        nonlocal frame, observation
        action = agent.select_action(observation, evaluation=True)
        new_observation, reward, done, _, info = env.step(action)
        observation = new_observation
        if done:
            observation, _ = env.reset()

        ax_rt1.clear()
        x = np.arange(len(slice_names))
        bars = ax_rt1.bar(x, info["slice_loads"], color=slice_colors, alpha=0.7)
        ax_rt1.set_xticks(x)
        ax_rt1.set_xticklabels(slice_names)
        ax_rt1.set_ylabel("Load (Mbps)")
        ax_rt1.set_title("Real-Time: Current Slice Load")

        for i, slice_info in enumerate(SLICES):
            ax_rt1.axhline(y=slice_info["bandwidth"],
                           xmin=i/len(slice_names),
                           xmax=(i+1)/len(slice_names),
                           color='r',
                           linestyle='--',
                           alpha=0.5)
        
        steps_history.append(frame)
        for i, load in enumerate(info["slice_loads"]):
            slice_load_history[i].append(load)
        max_history_points = 100  
        start_idx = max(0, len(steps_history) - max_history_points)
        visible_steps = steps_history[start_idx:]
        ax_rt2.clear()
        for i in range(len(slice_names)):
            ax_rt2.plot(visible_steps, slice_load_history[i][start_idx:],
                        label=slice_names[i], color=slice_colors[i])
        ax_rt2.set_xlim(max(0, frame - max_history_points), frame)
        ax_rt2.set_ylabel("Load (Mbps)")
        ax_rt2.set_xlabel("Step")
        ax_rt2.set_title("Real-Time: Slice Load History")
        ax_rt2.legend(loc='upper left')
        ax_rt2.grid(True)

        ax_rt3.clear()
        ue_x = []
        ue_y = []
        ue_colors = []
        ue_positions = {i: 0 for i in range(len(SLICES))}
        for ue in info["ues"]:
            slice_idx = ue["allocated_slice"]
            if slice_idx is None or slice_idx < 0:
                continue
            pos = ue_positions[slice_idx]
            ue_positions[slice_idx] += 1
            ue_x.append(slice_idx)
            ue_y.append(pos)
            ue_colors.append(ue["profile"])
        scatter_rt = ax_rt3.scatter(ue_x, ue_y, s=100, c=ue_colors, cmap='tab10',
                                      vmin=0, vmax=len(TRAFFIC_PROFILES)-1, edgecolors='k')
        ax_rt3.set_xticks(range(len(SLICES)))
        ax_rt3.set_xticklabels(slice_names)
        ax_rt3.set_ylabel("UE Count")
        ax_rt3.set_title("Real-Time: UE Allocation")
        ax_rt3.grid(True)
        
        ax_sim1.clear()
        for (p1, p2) in edges:
            ax_sim1.plot([p1[0], p2[0]], [p1[1], p2[1]],
                         color='gray', linestyle='--', linewidth=1)
        node_radius = 0.6
        for i, slice_info in enumerate(SLICES):
            x_pos, y_pos = node_positions[i]
            load_norm = min(info["slice_loads"][i] / slice_info["bandwidth"], 1.0)
            node_color = plt.cm.Reds(load_norm)
            circle = plt.Circle((x_pos, y_pos), node_radius, color=node_color,
                                ec='black', zorder=2)
            ax_sim1.add_patch(circle)
            ax_sim1.text(x_pos, y_pos+node_radius+0.2, slice_info["name"],
                         ha='center', va='bottom', fontsize=10, zorder=3)
        ue_x_net = []
        ue_y_net = []
        ue_colors_net = []
        for ue in info["ues"]:
            allocated_slice = ue.get("allocated_slice", -1)
            if allocated_slice is None or allocated_slice < 0:
                continue
            center_x, center_y = node_positions[allocated_slice]
            r = node_radius * math.sqrt(random.uniform(0, 1))
            theta = random.uniform(0, 2 * math.pi)
            pos_x = center_x + r * math.cos(theta)
            pos_y = center_y + r * math.sin(theta)
            ue_x_net.append(pos_x)
            ue_y_net.append(pos_y)
            ue_colors_net.append(ue["profile"])
        ax_sim1.scatter(ue_x_net, ue_y_net, c=ue_colors_net, cmap='tab10', s=100,
                        vmin=0, vmax=len(TRAFFIC_PROFILES)-1, edgecolors='k', zorder=3)
        ax_sim1.set_xlim(-radius-2, radius+2)
        ax_sim1.set_ylim(-radius-2, radius+2)
        ax_sim1.set_aspect('equal')
        ax_sim1.axis('off')

        ax_sim2.clear()
        ax_sim2.axis('off')
        metrics = f"Step: {frame}\nTotal UEs: {env.total_ues}\nActive UEs: {len(info['ues'])}"
        ax_sim2.text(0.5, 0.5, metrics, ha='center', va='center', fontsize=14)
        
        frame += 1
        return []
    
    ani = animation.FuncAnimation(fig, update, interval=update_interval, blit=False)
    plt.tight_layout()
    plt.show()
    return ani

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    env = NetworkSlicingEnv(arrival_rate=2)
    
    if not os.path.exists("models/sac_final.pth"):
        print("\nNo trained models found. Training agents...\n")
        
        num_episodes = 200  
        
        results = {}
        
        print("\nTraining DQN Agent...\n")
        dqn_agent, dqn_results = train_agent("dqn", env, num_episodes=num_episodes)
        results["DQN"] = dqn_results
        
        print("\nTraining SAC Agent...\n")
        sac_agent, sac_results = train_agent("sac", env, num_episodes=num_episodes)
        results["SAC"] = sac_results
        
        print("\nTraining PPO Agent...\n")
        ppo_agent, ppo_results = train_agent("ppo", env, num_episodes=num_episodes)
        results["PPO"] = ppo_results
        
        print("\nComparing agent performance...\n")
        plot_training_curves(results)
        
        best_agent_name = max(results.keys(), 
                             key=lambda k: results[k]["eval_rewards"][-1])
        
        print(f"\nBest performing agent: {best_agent_name}\n")
        
        if best_agent_name == "DQN":
            best_agent = dqn_agent
        elif best_agent_name == "SAC":
            best_agent = sac_agent
        else:
            best_agent = ppo_agent
            
    else:
        print("\nLoading pre-trained SAC agent...\n")
        best_agent = SACAgent(env.observation_space.shape[0], env.action_space.n)
        best_agent.actor.load_state_dict(torch.load("models/sac_final.pth"))

    print("\nRunning Combined Simulation...\n")
    run_combined_simulation(best_agent, env, sim_time=50, update_interval=100)
    
    print("Simulation complete!")

if __name__ == "__main__":
    main()
