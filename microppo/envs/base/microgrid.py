#
# Copyright (C) 2024 Daniel Ebi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import gym
import numpy as np
from abc import abstractmethod
from typing import List, Optional, Dict, Tuple

from microppo.envs.common.architectures import BasicMicroGridArchitecture


class BasicDecentralizedMicroGridEnv(gym.Env):

    def __init__(self, architecture: BasicMicroGridArchitecture, horizon: int):
        """
        Decentralized micro-grid with a basic architecture.

        :param BasicMicroGridArchitecture architecture: Micro-grid architecture (i.e., its components)
        :param int horizon: Time horizon.
        """
        self._architecture = architecture

        self.action_space = self.initialize_joint_action_space()
        self.observation_space = self.initialize_joint_state_space()

        self.log = ''
        self.reward = None
        self.cumulative_reward = 0
        self.n_steps = horizon
        self.horizon = horizon
        self.remaining_steps = self.horizon
        self.done = False

        self.pv_action: List[float, float] = None
        self.battery_action: List[float, float] = None
        self.revenue: float = None
        self.costs: float = None
        self.costs_battery_operation: float = None
        self.penalty: float = 0
        self.n_violations: int = 0

    @abstractmethod
    def initialize_joint_state_space(self):
        """
        Initialize the joint state space.

        :rtype: gym.spaces.Space
        :return: Joint state space.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_joint_action_space(self):
        """
        Initialize the joint action space.

        :rtype: gym.spaces.Space
        :return: Joint action space.
        """
        raise NotImplementedError

    def _reset(self):
        """
        Reset the micro-grid to its initial state.

        :rtype: np.array
        :return: Initial state of the micro-grid.
        """
        return np.array(self._architecture.reset(), dtype=np.float32)

    def _get_info(self) -> Dict[str, str]:
        """
        Return all relevant information about the micro-grid environment.

        :rtype: Dict[str, str]
        :return: Dictionary with all relevant information.
        """

        return {
            "load": self._architecture.household.state(),
            "power_generated": self._architecture.pv_system.state(),
            "energy_price": self._architecture.utility_grid.state(),
            "soc_battery": self._architecture.battery.state(),
            "pv_action": self.pv_action,
            "battery_action": self.battery_action,
            "reward": self.reward,
            "revenue": self.revenue,
            "costs": self.costs,
            "costs_battery_operation": self.costs_battery_operation,
            "penalty": self.penalty,
            "n_violations": self.n_violations,
            "remaining_steps": self.remaining_steps,
            "cumulative_reward": self.cumulative_reward,
            "capacity_pv": self._architecture.pv_system.capacity,
            "nominal_capacity_battery": self._architecture.battery.nominal_capacity
        }

    def _log_message(self) -> str:
        """
        Create log message.

        :rtype: str
        :return: Log message.
        """
        message = 'S:%s \tA_PV:%s \tA_B:%s \tRew:%s \tCum:%s \n' % (
            self.horizon - self.remaining_steps, self.pv_action, self.battery_action, self.reward,
            self.cumulative_reward)
        return message

    def step(self, action) -> Tuple[List, float, bool, dict]:
        """
        Perform the joint action chosen by the agent on the environment.

        :param action: Joint action.

        :rtype:  Tuple[List, float, bool, dict]
        :return: Next state, reward, whether the episode is finished or not, dictionary containing information about the environment.
        """
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        """
        Reset the environment to its initial state.

        :param int seed: Seed for random processes (optional, default is None).
        :param bool return_info: Whether to return information about environment or not (optional, default is None).
        :param dict options: Options (optional, default is None).

        :rtype: List[Union[str, int, float, bool]]
        :return: Initial state of the mirco-grid.
        """
        self.log = ''
        self.remaining_steps = self.horizon
        self.pv_action = None
        self.battery_action = None
        self.reward = None
        self.revenue = None
        self.costs = None
        self.costs_battery_operation = None
        self.penalty = 0
        self.n_violations = 0
        self.cumulative_reward = 0
        self.done = False

        observation = self._reset()
        return observation

    def render(self, mode="human"):
        """
        Render information about the environment (i.e., print the log message).

        :param str mode: Rendering mode (optional, default is "human").
        """
        self.log = self._log_message()
        print(self.log)

    def close(self):
        """
        Close the environment.
        """
        pass

    def print_last_log(self):
        """
        Print the last log message.
        """
        print(self.log.split('\n')[-2])

    @property
    def architecture(self):
        """
        Return the architecture of the micro-grid.

        :rtype: BasicMicroGridArchitecture
        :return: Micro-grid architecture.
        """
        return self._architecture
