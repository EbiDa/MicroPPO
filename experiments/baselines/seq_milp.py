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
from pyomo.environ import *
from typing import List, Optional

from microppo.envs.common.architectures import BasicMicroGridArchitecture


class SequentialMicroGridMILP:
    def __init__(self):
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

        observation = self.architecture.reset()
        return observation

    def _build(self) -> Model:
        household = self.architecture.household
        pv_system = self.architecture.pv_system
        battery = self.architecture.battery
        utility_grid = self.architecture.utility_grid
        discount_factor = self.architecture.utility_grid.export_price_discount

        opt_model = ConcreteModel()

        # Decision variables
        opt_model.grid_to_household = Var(within=NonNegativeReals, initialize=0)
        opt_model.grid_to_battery = Var(within=NonNegativeReals, initialize=0)
        opt_model.pv_to_household = Var(within=NonNegativeReals, initialize=0)
        opt_model.pv_to_battery = Var(within=NonNegativeReals, initialize=0)
        opt_model.pv_to_grid = Var(within=NonNegativeReals, initialize=0)
        opt_model.battery_to_household = Var(within=NonNegativeReals, initialize=0)
        opt_model.battery_to_grid = Var(within=NonNegativeReals, initialize=0)
        opt_model.y_charge = Var(domain=Boolean, initialize=0)
        opt_model.y_discharge = Var(domain=Boolean, initialize=0)

        # Objective function
        opt_model.obj = Objective(expr=utility_grid.state() * discount_factor * (
                opt_model.pv_to_grid + (
                opt_model.battery_to_grid * battery.discharging_efficiency)) - utility_grid.state() * (
                                               opt_model.grid_to_household + opt_model.grid_to_battery) - battery.costs_of_operation * (
                                               opt_model.pv_to_battery + opt_model.grid_to_battery + opt_model.battery_to_household + opt_model.battery_to_grid),
                                  sense=maximize)

        # Constraints
        opt_model.constraints = ConstraintList()
        opt_model.constraints.add(
            expr=opt_model.pv_to_household + opt_model.pv_to_battery + opt_model.pv_to_grid == pv_system.state())
        opt_model.constraints.add(
            expr=opt_model.pv_to_household + (
                    opt_model.battery_to_household * battery.discharging_efficiency) + opt_model.grid_to_household == household.state())
        opt_model.constraints.add(
            expr=opt_model.pv_to_battery + opt_model.grid_to_battery <= min(battery.available_upward_power(),
                                                                            battery.max_charging_rate) * opt_model.y_charge)
        opt_model.constraints.add(expr=opt_model.battery_to_household + opt_model.battery_to_grid <= min(
            battery.available_downward_power(), battery.max_discharging_rate) * opt_model.y_discharge)

        opt_model.constraints.add(expr=opt_model.y_charge + opt_model.y_discharge <= 1)
        return opt_model

    def step(self, action):
        if self.done:
            return None
        else:
            self.grid_action = action[:2]
            self.pv_action = action[2:-2]
            self.battery_action = action[-2:]

            total_battery_discharging_power: float = self.battery_action[0] + self.battery_action[1]
            total_battery_charging_power: float = self.pv_action[1] + self.grid_action[1]
            power_from_grid: float = self.grid_action[0] + self.grid_action[1]
            power_to_grid: float = self.pv_action[2] + (
                    self.battery_action[1] * self.architecture.battery.discharging_efficiency)

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

    def solve(self) -> List[float]:
        instance = self._build()
        # Solver
        solver = SolverFactory('glpk')
        # Solve the optimization problem
        results = solver.solve(instance)
        action = [float(instance.grid_to_household.value), float(instance.grid_to_battery.value),
                  float(instance.pv_to_household.value), float(instance.pv_to_battery.value),
                  float(instance.pv_to_grid.value), float(instance.battery_to_household.value),
                  float(instance.battery_to_grid.value)]
        return action
