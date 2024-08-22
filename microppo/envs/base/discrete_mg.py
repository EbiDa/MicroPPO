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
import itertools
import numpy as np
from typing import List, Callable, Union, Optional

from microppo.envs.common.architectures import BasicMicroGridArchitecture
from .microgrid import BasicDecentralizedMicroGridEnv


class DiscreteDecentralizedMicroGridEnv(BasicDecentralizedMicroGridEnv):

    def __init__(self, architecture: BasicMicroGridArchitecture, horizon: int,
                 action_space_masking: Optional[Callable] = None):
        """
        Decentralized micro-grid with a basic architecture, but featuring a discrete action space.

        :param BasicMicroGridArchitecture architecture: Micro-grid architecture (i.e., its components)
        :param int horizon: Time horizon.
        :param Callable action_space_masking: Action space mask (optional).
       """
        self.action_space_masking: Callable = action_space_masking
        self.multi_action_mapping = ()

        super().__init__(architecture=architecture, horizon=horizon)

    def initialize_joint_state_space(self):
        # Continuous state space
        obs_space_lower_bounds = list(itertools.chain(
            *[component.state_space.lower_bound for component in self._architecture.all_components()]))
        obs_space_upper_bounds = list(itertools.chain(
            *[component.state_space.upper_bound for component in self._architecture.all_components()]))
        return gym.spaces.Box(low=np.float32(obs_space_lower_bounds),
                              high=np.float32(obs_space_upper_bounds), dtype=np.float32)

    def initialize_joint_action_space(self):
        # Discrete action space
        action_subspaces = tuple(
            [component.action_space.n_options for component in self._architecture.controllable_components() if
             component.action_space is not None])
        self.multi_action_mapping = tuple(np.ndindex(action_subspaces))

        # Action masking
        if self.action_space_masking is None:
            n_actions = np.prod(action_subspaces)
        else:
            is_allowed_action = self.action_space_masking(actions=self.multi_action_mapping)
            n_not_allowed_actions = is_allowed_action.count(False)
            self.multi_action_mapping = tuple(itertools.compress(list(self.multi_action_mapping), is_allowed_action))
            n_actions = np.prod(action_subspaces) - n_not_allowed_actions

        return gym.spaces.Discrete(n=n_actions)

    def step(self, action: Union[List, np.ndarray]):
        household = self._architecture.household
        pv_system = self._architecture.pv_system
        battery = self._architecture.battery
        utility_grid = self._architecture.utility_grid
        calendar = self._architecture.calendar

        if self.done:
            return None
        else:
            load_deficit = - household.state()
            charging_power_from_pv = 0
            charging_power_from_grid = 0
            power_from_grid = 0
            power_to_grid = 0
            actual_total_charging_power = 0

            action = int(action)

            # Obtain sub-actions from joint action
            self.pv_action = [self.multi_action_mapping[int(action)][0],
                              self._architecture.pv_system.action_space.value(
                                  option=self.multi_action_mapping[action][0])]
            self.battery_action = [self.multi_action_mapping[int(action)][1],
                                   self._architecture.battery.action_space.value(
                                       option=self.multi_action_mapping[action][1])]
            pv_action_type = self.pv_action[0]
            pv_power_share = self.pv_action[1]
            battery_action_type = self.battery_action[0]
            battery_power_share = self.battery_action[1]

            # Reserved power generated by the PV system
            allocated_pv_power = pv_system.state() * pv_power_share

            # PV power -> local loads
            if pv_action_type < 10:
                load_deficit += allocated_pv_power
            # PV power -> battery
            elif pv_action_type < 20:
                charging_power_from_pv += allocated_pv_power
            # PV power -> utility grid
            else:
                power_to_grid += allocated_pv_power

            # Sell excess power to the utility grid
            power_to_grid += pv_system.state() - allocated_pv_power

            total_battery_charging_power = charging_power_from_pv
            total_battery_discharging_power = 0

            # Battery -> local loads
            if battery_action_type < 10:
                discharging_power_household = battery.discharge(weight=battery_power_share)
                total_battery_discharging_power += discharging_power_household
                load_deficit += discharging_power_household * battery.discharging_efficiency

            # Battery -> utility grid
            elif battery_action_type < 20:
                discharging_power_grid = self._architecture.battery.discharge(weight=battery_power_share)
                total_battery_discharging_power += discharging_power_grid
                power_to_grid += discharging_power_grid * battery.discharging_efficiency

            # Utility grid -> battery
            elif 20 <= battery_action_type < 30:
                charging_power_from_grid = battery_power_share * battery.available_upward_power()
                total_battery_charging_power += charging_power_from_grid

            # Ensure that battery is not over-charged
            if total_battery_charging_power > 0:
                actual_total_charging_power += battery.charge(weight=1.0,
                                                              input_power=total_battery_charging_power) / battery.charging_efficiency
                power_from_grid += (
                                           charging_power_from_grid / total_battery_charging_power) * actual_total_charging_power

            power_from_grid += - min(load_deficit, 0)
            power_to_grid += max(load_deficit, 0)

            # Revenue obtained by selling power to the utility grid
            revenue = power_to_grid * utility_grid.state() * utility_grid.export_price_discount
            self.revenue = revenue

            # Costs for buying power from the utility grid
            costs = power_from_grid * utility_grid.state()
            self.costs = costs

            # Costs for charging / discharging the battery
            costs_battery_operation = battery.costs_of_operation * (
                    (actual_total_charging_power * battery.charging_efficiency) + total_battery_discharging_power)
            self.costs_battery_operation = costs_battery_operation

            # Reward and soft penalty
            self.reward = revenue - costs - costs_battery_operation
            self.penalty = 0
            self.reward -= self.penalty

            # Update the cumulative reward
            self.cumulative_reward += self.reward

            self.remaining_steps -= 1
            if self.remaining_steps == 0:
                self.done = True

            # Logging
            info = self._get_info()
            self.log += self._log_message()

            # Transition to the next state depending on the selected action
            next_state = self._architecture.step(battery_power_in=actual_total_charging_power,
                                                 battery_power_out=total_battery_discharging_power)

            return next_state, self.reward, self.done, info
