import numpy as np
from gym import spaces
from torch import nn


class Brain():
    def __init__(self, action_space, observation_channel):
        self.inner_state_num = 128
        self.inner_state = None
        self.obs_network = nn.Sequential(
            nn.Conv2d(observation_channel, 8, (3, 4)),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=2),
            nn.ReLU(),

        )
        self.action_network = nn.Sequential(
            nn.Linear(self.inner_state_num, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_space),
            nn.ReLU(),
        )
        self.critic_network = nn.Sequential(
            nn.Linear(self.inner_state_num, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.ReLU(),
        )

    def observe(self, obs):
        self.inner_state = self.obs_network(obs)

    def action(self, obs):
        self.observe(obs)
        return self.action_network(self.inner_state), self.critic_network(self.inner_state)


class MahjongBot():
    def __init__(self):
        self.action_space = spaces.Discrete(34+6)  # 34种牌，（吃,碰,杠,立直,胡,放弃）6种动作
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(34), dtype=np.int8)  # 34种牌，4个玩家，4（手牌）+3+3+3（三个对手）+4（牌河）+1（宝牌）=18种状态
        self.brain = Brain(self.action_space.n, self.observation_space.n)

    def obs(self, obs):
        self.brain.observe(obs)

    def action(self, obs):
        return self.brain.action(obs)
