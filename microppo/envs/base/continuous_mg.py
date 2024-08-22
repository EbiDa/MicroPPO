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
from typing import Union, List

from microppo.envs.common.architectures import BasicMicroGridArchitecture
from .microgrid import BasicDecentralizedMicroGridEnv


class ContinuousDecentralizedMicroGridEnv(BasicDecentralizedMicroGridEnv):
    def __init__(self, architecture: BasicMicroGridArchitecture, horizon: int):
        """
        Decentralized micro-grid with a basic architecture, but featuring a continuous action space.

        :param BasicMicroGridArchitecture architecture: Micro-grid architecture (i.e., its components)
        :param int horizon: Time horizon.
        """
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
        # Continuous action space
        action_space_lower_bounds = list(itertools.chain(
            *[component.action_space.lower_bound for component in
              self._architecture.controllable_components()]))
        action_space_upper_bounds = list(itertools.chain(
            *[component.action_space.upper_bound for component in
              self._architecture.controllable_components()]))
        return gym.spaces.Box(low=np.float32(action_space_lower_bounds),
                              high=np.float32(action_space_upper_bounds), dtype=np.float32)

    def step(self, action: Union[List, np.ndarray]):
        state = self._architecture.state()
        household = self._architecture.household
        pv_system = self._architecture.pv_system
        battery = self._architecture.battery
        utility_grid = self._architecture.utility_grid
        calendar = self._architecture.calendar

        # Apply rounding
        action = action.round(4)

        if self.done:
            return state, None, self.reward, self.done, {}
        else:
            load_deficit = - household.state()
            charging_power_from_pv = 0
            charging_power_from_grid = 0
            power_from_grid = 0
            power_to_grid = 0

            # Obtain sub-actions from joint action
            self.pv_action = [action[0], max(0, action[1])]
            self.battery_action = [min(0, action[1]), action[2]]

            # PV system related actions (A)
            # (A1) PV power -> local loads
            pv_to_household = self.pv_action[0] * pv_system.state()
            # (A2) PV power -> battery
            pv_to_battery = self.pv_action[1] * pv_system.state()
            # (A3) PV power -> utility grid
            pv_to_grid = (1 - self.pv_action[0] - self.pv_action[1]) * pv_system.state()

            load_deficit += pv_to_household
            charging_power_from_pv += pv_to_battery
            power_to_grid += pv_to_grid

            # Battery related actions (B)
            total_battery_charging_power = charging_power_from_pv
            total_battery_discharging_power = 0
            battery_to_household = 0
            battery_to_grid = 0

            # (B1) Battery -> local loads
            if self.battery_action[0] < 0:
                battery_to_household = battery.discharge(
                    weight=abs(self.battery_action[0]))
                load_deficit += battery_to_household * battery.discharging_efficiency

            # (B2) Battery -> utility grid
            if self.battery_action[1] < 0:
                battery_to_grid = battery.discharge(weight=abs(self.battery_action[1]))
                power_to_grid += battery_to_grid * battery.discharging_efficiency

            total_battery_discharging_power += battery_to_household + battery_to_grid

            # (B3) Utility grid -> battery
            if self.battery_action[1] > 0:
                charging_power_from_grid = self.battery_action[1] * battery.available_upward_power()
                total_battery_charging_power += charging_power_from_grid

            # Ensure that battery is not over-charged
            actual_total_charging_power = 0
            if total_battery_charging_power > 0:
                actual_total_charging_power += battery.charge(weight=1.0,
                                                              input_power=total_battery_charging_power) / battery.charging_efficiency
                if actual_total_charging_power < total_battery_charging_power:
                    if actual_total_charging_power < charging_power_from_pv:
                        power_to_grid += charging_power_from_pv - actual_total_charging_power
                        power_from_grid += 0
                    else:
                        power_from_grid += actual_total_charging_power - charging_power_from_pv
                else:
                    power_from_grid += actual_total_charging_power - charging_power_from_pv

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
                    (total_battery_charging_power * battery.charging_efficiency) + total_battery_discharging_power)
            self.costs_battery_operation = costs_battery_operation

            # Reward
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


class PenalizedContinuousDecentralizedMicroGridEnv(ContinuousDecentralizedMicroGridEnv):
    def __init__(self, architecture: BasicMicroGridArchitecture, horizon: int):
        """
        Decentralized micro-grid with a basic architecture, but featuring a continuous action space and implementing a
        soft penalty in the reward function.

        :param BasicMicroGridArchitecture architecture: Micro-grid architecture (i.e., its components)
        :param int horizon: Time horizon.
        """
        super().__init__(architecture=architecture, horizon=horizon)

    def step(self, action: Union[List, np.ndarray]):
        state = self._architecture.state()
        household = self._architecture.household
        pv_system = self._architecture.pv_system
        battery = self._architecture.battery
        utility_grid = self._architecture.utility_grid
        calendar = self._architecture.calendar

        if self.done:
            return state, None, self.reward, self.done, {}
        else:
            load_deficit = - household.state()
            charging_power_from_pv = 0
            charging_power_from_grid = 0
            power_from_grid = 0
            power_to_grid = 0
            penalized_power_pv = 0
            penalized_power_battery = 0
            pen_sim = 0

            # Obtain sub-actions from joint action
            self.pv_action = [round(action[0], 4), round(action[1], 4)]
            self.battery_action = [round(action[2], 4), round(action[3], 4), round(action[4], 4)]

            # PV system related actions (A)
            # (A1) PV power -> local loads
            pv_to_household = self.pv_action[0] * pv_system.state()
            # (A2) PV power -> battery
            pv_to_battery = self.pv_action[1] * pv_system.state()
            # (A3) PV power -> utility grid
            pv_to_grid = (1 - self.pv_action[0] - self.pv_action[1]) * pv_system.state()

            load_deficit += pv_to_household
            charging_power_from_pv += pv_to_battery
            power_to_grid += pv_to_grid

            # Compute the overshoot in case of excessive use of the generated power
            if sum(self.pv_action) > 1:
                penalized_power_pv += np.abs(1 - sum(self.pv_action)) * pv_system.state()

            # Battery related actions (B)
            total_battery_charging_power = charging_power_from_pv
            total_battery_discharging_power = 0

            # Add penalty in case the battery is charged and discharged simultaneously
            if not ((self.battery_action[0] >= 0 and self.battery_action[1] >= 0) or (
                    self.battery_action[0] <= 0 and self.battery_action[1] <= 0)):
                pen_sim += 1

            # Compute the overshoot in case of excessive use of battery discharging
            if sum(self.battery_action[:-1]) > 1:
                penalized_power_battery += np.abs(sum(self.battery_action[:-1]) - 1) * max(
                    battery.available_downward_power(),
                    battery.max_discharging_rate)

            # (B1) Battery -> local loads
            battery_to_household = battery.discharge(weight=self.battery_action[0]) * battery.discharging_efficiency
            load_deficit += battery_to_household

            # (B2) Battery -> utility grid
            battery_to_grid = battery.discharge(weight=self.battery_action[1])
            power_to_grid += battery_to_grid * battery.discharging_efficiency

            total_battery_discharging_power += battery_to_household + battery_to_grid

            # (B3) Utility grid -> battery
            charging_power_from_grid = self.battery_action[2] * battery.available_upward_power()
            total_battery_charging_power += charging_power_from_grid

            actual_total_charging_power = 0

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
                    (total_battery_charging_power * battery.charging_efficiency) + total_battery_discharging_power)
            self.costs_battery_operation = costs_battery_operation

            # Reward and soft penalty
            self.reward = revenue - costs - costs_battery_operation
            self.penalty = (10 * penalized_power_pv) + (5 * penalized_power_battery)
            self.reward -= self.penalty

            # Update the cumulative reward
            self.cumulative_reward += self.reward

            # Update the number of violations if an infeasible action was selected
            if penalized_power_pv > 0 or penalized_power_battery > 0 or pen_sim > 0:
                self.n_violations += 1

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
