import numpy as np

class Box:
    def __init__(self, low, high):
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.ndim = len(low)
    @property
    def shape(self):
        return (self.ndim,)
    def sample(self):
        return np.random.uniform(self.low, self.high)

class NDMaze:
    def __init__(self, ndim: int = 2, maze_size: float = 5.0, goal_size: float = 1.0, max_steps: int = 200):
        self.dt = 0.1
        self.ndim = ndim
        self.maze_size = maze_size
        self.goal_size = goal_size
        self.max_steps = max_steps
        self.num_steps = 0

        if not goal_size * 2 < maze_size * 0.5:
            raise ValueError(f"0.5*maze_size={0.5*maze_size} must be greater than 2*goal_size={2*goal_size}")

        pos_range = 2 * maze_size * np.ones(ndim, dtype=np.float32)
        vel_range = 2 * maze_size * np.ones(ndim, dtype=np.float32)
        obs_range = np.concatenate((pos_range, vel_range))
        self.observation_space = Box(-obs_range, obs_range)
        self.action_space = Box(-np.ones(ndim, dtype=np.float32), np.ones(ndim, dtype=np.float32))
        self.observation = np.zeros(2 * ndim, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.num_steps = 0
        while True:
            self.observation = self.observation_space.sample()
            if self.goal_size * 2.0 <= self.distance <= self.maze_size * 0.5:
                break
        self.velocity = 0.0
        return self.observation.copy(), {}

    def step(self, action):
        self.position += self.dt * self.velocity
        self.velocity += self.dt * self.action_to_force(action)
        self.num_steps += 1
        current_distance = self.distance
        reward = 0.0
        terminated = False
        if current_distance > self.maze_size:
            reward = -10.0
            terminated = True
        elif current_distance < self.goal_size:
            reward = 10.0
            terminated = True
        truncated = self.num_steps >= self.max_steps
        return self.observation.copy(), reward, terminated, truncated, {}

    def action_to_force(self, action):
        action = np.array(action, dtype=np.float32)
        norm = np.linalg.norm(action)
        action = action if norm <= 1 else action / norm
        return action

    @property
    def distance(self) -> float:
        return np.linalg.norm(self.position, ord=2)

    @property
    def position(self) -> np.ndarray:
        return self.observation[:self.ndim]

    @position.setter
    def position(self, value: ArrayLike):
        self.observation[:self.ndim] = value

    @property
    def velocity(self) -> np.ndarray:
        return self.observation[self.ndim:]

    @velocity.setter
    def velocity(self, value: ArrayLike):
        self.observation[self.ndim:] = value