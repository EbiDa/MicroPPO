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
import numpy as np
import pandas as pd
from pyomo.environ import *
from typing import Dict, List, Optional

from experiments.config import PATH_TO_GLPK_EXECUTABLE
from microppo.envs.common.architectures import BasicMicroGridArchitecture


class RuleBasedApproach:
    def __init__(self, policy: str):
        self.policy: str = policy

        self._architecture: BasicMicroGridArchitecture = None
        self.horizon: int = None

        self.reward: float = None
        self.cumulative_reward: float = 0
        self.n_steps: int = self.horizon
        self.remaining_steps: int = self.horizon
        self.done: bool = False

        self.grid_action: List[float, float] = None
        self.pv_action: List[float, float] = None
        self.battery_action: List[float, float] = None
        self.revenue: float = None
        self.costs: float = None
        self.costs_battery_operation: float = None
        self.penalty: float = None
        self.n_violations: int = None
        self._seen_energy_prices: List[float] = []

    def env(self, architecture: BasicMicroGridArchitecture, horizon: int):
        self._architecture = architecture
        self.horizon = horizon

        self.reset()

    @property
    def architecture(self) -> BasicMicroGridArchitecture:
        return self._architecture

    def _get_info(self):
        return [{
            "load": self.architecture.household.state(),
            "power_generated": self.architecture.pv_system.state(),
            "energy_price": self.architecture.utility_grid.state(),
            "soc_battery": self.architecture.battery.state(),
            "grid_action": self.grid_action,
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
            "capacity_pv": self.architecture.pv_system.capacity,
            "nominal_capacity_battery": self.architecture.battery.nominal_capacity
        }]

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
        self._seen_energy_prices = []

        observation = self.architecture.reset()
        return observation

    def step(self, action):
        if self.done:
            return None
        else:
            self.battery_action = action[:2]
            self.grid_action = action[2:]
            self.pv_action = []

            total_battery_charging_power: float = self.battery_action[0]
            total_battery_discharging_power: float = self.battery_action[1]
            power_from_grid: float = self.grid_action[0]
            power_to_grid: float = self.grid_action[1]

            revenue = power_to_grid * self.architecture.utility_grid.state() * self.architecture \
                .utility_grid.export_price_discount
            costs = power_from_grid * self.architecture.utility_grid.state()
            costs_battery_operation = self.architecture.battery.costs_of_operation * (
                    total_battery_charging_power + total_battery_discharging_power)

            self.revenue = revenue
            self.costs = costs
            self.costs_battery_operation = costs_battery_operation

            self.reward = revenue - costs - costs_battery_operation

            self.penalty = 0
            self.cumulative_reward += self.reward

            self.remaining_steps -= 1
            if self.remaining_steps == 0:
                self.done = True

            info = self._get_info()
            observation = self.architecture.step(battery_power_in=total_battery_charging_power,
                                                 battery_power_out=total_battery_discharging_power)

            return observation, self.reward, self.done, info

    def apply_rules(self) -> List[float]:
        household = self.architecture.household
        pv_panel = self.architecture.pv_system
        battery = self.architecture.battery
        utility_grid = self.architecture.utility_grid

        load_difference = - household.state()
        charging_power = 0
        discharging_power = 0
        power_from_grid = 0
        power_to_grid = 0

        energy_price = utility_grid.state()
        power_from_pv = pv_panel.state()
        power_battery_out = battery.available_downward_power() * battery.discharging_efficiency
        power_battery_in = battery.available_upward_power() / battery.charging_efficiency

        if self.policy == "economic":
            if pv_panel.state() >= household.state():
                load_difference = 0
                overproduced_power = pv_panel.state() - household.state()

                if energy_price > np.median(self._seen_energy_prices):
                    # or power_battery_in < overproduced_power:
                    power_to_grid += overproduced_power
                else:
                    tmp_charging_power = battery.charge(weight=1, input_power=overproduced_power)
                    overproduced_power -= tmp_charging_power / battery.charging_efficiency
                    charging_power += tmp_charging_power
                    if overproduced_power > 0:
                        power_to_grid += overproduced_power
                power_from_pv = 0

            else:
                if power_battery_out > 0:  # household.state():
                    weight = household.state() / power_battery_out
                    tmp_discharging_power = battery.discharge(weight=weight)
                    discharging_power += tmp_discharging_power
                    load_difference += tmp_discharging_power * battery.discharging_efficiency
                load_difference += power_from_pv
                power_from_pv = 0

            power_from_grid += - min(0, load_difference)
            power_to_grid += max(0, load_difference)

            self._seen_energy_prices.append(energy_price)

            return [charging_power, discharging_power, power_from_grid, power_to_grid]

        elif self.policy == "own_consumption":
            if pv_panel.state() >= household.state():
                load_difference = 0
                overproduced_power = pv_panel.state() - household.state()

                tmp_charging_power = battery.charge(weight=1, input_power=overproduced_power)
                overproduced_power -= tmp_charging_power / battery.charging_efficiency
                charging_power += tmp_charging_power

                if overproduced_power > 0:
                    power_to_grid += overproduced_power

                power_from_pv = 0

            else:
                if power_battery_out > 0:  # household.state():
                    weight = household.state() / power_battery_out
                    tmp_discharging_power = battery.discharge(weight=weight)
                    discharging_power += tmp_discharging_power
                    load_difference += tmp_discharging_power * battery.discharging_efficiency
                load_difference += power_from_pv
                power_from_pv = 0

            power_from_grid += - min(0, load_difference)
            power_to_grid += max(0, load_difference)

            self._seen_energy_prices.append(energy_price)

            return [charging_power, discharging_power, power_from_grid, power_to_grid]

        else:
            raise ValueError(self.policy)
