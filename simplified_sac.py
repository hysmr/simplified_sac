import torch
import numpy as np

def build_mlp(dims: tuple[int, ...]):
    l = [torch.nn.Linear(dims[0], dims[1])]
    for i in range(1, len(dims)-1):
        l.append(torch.nn.ReLU())
        l.append(torch.nn.Linear(dims[i], dims[i+1]))
    return torch.nn.Sequential(*l)

class DoubleQ(torch.nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.q1 = build_mlp((observation_dim + action_dim, *hidden_dims, 1))
        self.q2 = build_mlp((observation_dim + action_dim, *hidden_dims, 1))
    def forward(self, observation, action):
        z = torch.cat((observation, action), dim=-1)
        return self.q1(z).flatten(), self.q2(z).flatten()
    def clipped(self, observation, action):
        q1, q2 = self(observation, action)
        return torch.min(q1, q2)
        
class Critic(torch.nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.online = DoubleQ(observation_dim, action_dim, hidden_dims)
        self.target = DoubleQ(observation_dim, action_dim, hidden_dims)
        self.target.load_state_dict(self.online.state_dict())
        self.target.train(False)
    def forward(self, observation, action):
        return self.online(observation, action), self.target(observation, action)
    @torch.no_grad()
    def update(self, tau):
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data.copy_(tau * param_online.data + (1 - tau) * param_target.data)

class Actor(torch.nn.Module):
    def __init__(self, observation_dim: int, action_dim:int , hidden_dims: tuple[int, ...]):
        super().__init__()
        self.network = build_mlp((observation_dim, *hidden_dims, action_dim * 2))
    def forward(self, observation):
        mean, logstd = self.network(observation).hsplit(2)
        logstd = torch.clamp(logstd, -20, 2)
        normal = torch.distributions.normal.Normal(mean, logstd.exp())
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = (normal.log_prob(z) - (1 - action**2 + 1e-6).log()).sum(dim=1)
        return action, log_prob, mean

class Policy(torch.nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims, action_scale, action_bias):
        super().__init__()
        self.actor = Actor(observation_dim, action_dim, hidden_dims)
        self.critic = Critic(observation_dim, action_dim, hidden_dims)
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('target_entropy', torch.tensor(-action_dim, dtype=torch.float32))
        self.register_buffer('action_scale', torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor(action_bias, dtype=torch.float32))
        self.register_buffer('observation_dim', torch.tensor(observation_dim, dtype=torch.long))
        self.register_buffer('action_dim', torch.tensor(action_dim, dtype=torch.long))
        self.register_buffer('hidden_dims', torch.tensor(hidden_dims, dtype=torch.long))
    @torch.no_grad()
    def predict(self, observation, deterministic=False):
        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(-1, self.observation_dim)
        action, log_prob, mean = self.actor(observation)
        action = self.action_scale * action + self.action_bias
        mean = self.action_scale * mean + self.action_bias
        return action.squeeze(0).numpy() if not deterministic else mean.squeeze(0).numpy()
    def compute_loss(self, observations, actions, next_observations, rewards, discounts):
        actions = (actions - self.action_bias) / self.action_scale
        alpha = self.log_alpha.exp().item()
        with torch.no_grad():
            next_actions_pi, next_log_probs_pi, _ = self.actor(next_observations)
            next_q_values = self.critic.target.clipped(next_observations, next_actions_pi)
            target_q_values = rewards + discounts * (next_q_values - alpha * next_log_probs_pi)
        q1, q2 = self.critic.online(observations, actions)
        q1_loss = torch.nn.functional.mse_loss(q1, target_q_values)
        q2_loss = torch.nn.functional.mse_loss(q2, target_q_values)
        critic_loss =  0.5 * (q1_loss + q2_loss)
        actions_pi, log_probs_pi, _ = self.actor(observations)
        q_values_pi = self.critic.online.clipped(observations, actions_pi)
        actor_loss = (alpha * log_probs_pi - q_values_pi).mean()
        alpha_loss = -(self.log_alpha * (log_probs_pi + self.target_entropy).detach()).mean()        
        return actor_loss, critic_loss, alpha_loss

class Optims(object):
    def __init__(self, policy):
        self.optim_actor = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
        self.optim_critic = torch.optim.Adam(policy.critic.parameters(), lr=3e-4)
        self.optim_alpha = torch.optim.Adam([policy.log_alpha], lr=3e-4)
    def step(self, actor_loss, critic_loss, alpha_loss):
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()
        
class ReplayBuffer(object):    
    def __init__(self, buffer_size: int, observation_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.head = 0
        self.full = False
        self.observations = np.zeros((buffer_size, observation_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, observation_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.discounts = np.zeros(buffer_size, dtype=np.float32)
    def clear(self) -> None:
        self.head = 0
        self.full = False
    def add(self, observation: np.ndarray, action: np.ndarray, next_observation: np.ndarray, reward: float, discount: float) -> None:
        self.observations[self.head][:] = observation.flatten()
        self.actions[self.head][:] = action.flatten()
        self.next_observations[self.head][:] = next_observation
        self.rewards[self.head] = reward
        self.discounts[self.head] = discount
        self.head += 1
        if self.head == self.buffer_size:
            self.head = 0
            self.full = True
    def last(self) -> int:
        return self.buffer_size if self.full else self.head
    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.last(), size=batch_size)
        return (torch.as_tensor(self.observations[indices], dtype=torch.float32),
                torch.as_tensor(self.actions[indices], dtype=torch.float32),
                torch.as_tensor(self.next_observations[indices], dtype=torch.float32),
                torch.as_tensor(self.rewards[indices], dtype=torch.float32),
                torch.as_tensor(self.discounts[indices], dtype=torch.float32))

class Sampler(object):
    def __init__(self):
        self.last_observation = None
        self.need_reset = True
    def step(self, env, policy, buffer, gamma):
        if self.need_reset:
            self.last_observation, info = env.reset()
            self.need_reset = False
        action = policy.predict(self.last_observation, deterministic=False)
        next_observation, reward, terminated, truncated, info = env.step(action)
        discount = gamma * (0.0 if terminated else 1.0)
        buffer.add(self.last_observation, action, next_observation, reward, discount)
        self.need_reset = terminated or truncated
        self.last_observation = next_observation

def build(env, buffer_size:int = 1000000, hidden_dims: tuple[int, ...] = (256, 256)):
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    low, high = env.action_space.low, env.action_space.high
    scale = 0.5 * (high - low)
    bias = 0.5 * (high + low)
    policy = Policy(observation_dim, action_dim, hidden_dims, scale, bias)
    optims = Optims(policy)
    buffer = ReplayBuffer(buffer_size, observation_dim, action_dim)
    return policy, optims, buffer

def train(env, policy, optims, buffer, total_timesteps: int, learning_starts: int=100, batch_size: int=256, gamma: float=0.99, tau: float=0.005, verbose: int=1):
    sampler = Sampler()
    timestep = 0
    while timestep < total_timesteps:
        sampler.step(env, policy, buffer, gamma)
        batch = buffer.sample(batch_size)
        if learning_starts < timestep:
            actor_loss, critic_loss, alpha_loss = policy.compute_loss(*batch)
            optims.step(actor_loss, critic_loss, alpha_loss)
            policy.critic.update(tau)
            if verbose == 1 and timestep % 1000 == 0:
                print(f"{timestep}/{total_timesteps}. actor_loss={actor_loss}, critic_loss={critic_loss}, alpha_loss={alpha_loss}")
        timestep += 1

def save(path, policy, optims, buffer):
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'optim_actor_state_dict': optims.optim_actor.state_dict(),
        'optim_critic_state_dict': optims.optim_critic.state_dict(),
        'optim_alpha_state_dict': optims.optim_alpha.state_dict(),
        'buffer': buffer,  # Assumes ReplayBuffer is pickleable.
    }
    torch.save(checkpoint, path)

def load(path):
    checkpoint = torch.load(path)
    policy_state = checkpoint['policy_state_dict']
    
    observation_dim = int(policy_state['observation_dim'].item())
    action_dim = int(policy_state['action_dim'].item())
    hidden_dims = tuple(policy_state['hidden_dims'].tolist())
    action_scale = policy_state['action_scale']
    action_bias = policy_state['action_bias']
    
    policy = Policy(observation_dim, action_dim, hidden_dims, action_scale, action_bias)
    optims = Optims(policy)
    
    policy.load_state_dict(policy_state)
    optims.optim_actor.load_state_dict(checkpoint['optim_actor_state_dict'])
    optims.optim_critic.load_state_dict(checkpoint['optim_critic_state_dict'])
    optims.optim_alpha.load_state_dict(checkpoint['optim_alpha_state_dict'])
    
    buffer = checkpoint['buffer']
    
    return policy, optims, buffer
        