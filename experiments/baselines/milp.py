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
import pandas as pd
from pyomo.environ import *
from typing import Dict

from experiments.config import PATH_TO_GLPK_EXECUTABLE
from microppo.envs.common.architectures import BasicMicroGridArchitecture
from microppo.envs.common.components import PVSystem, Battery


class MicroGridMILP:
    def __init__(self):
        self.horizon: int = None
        self.pv_system: PVSystem = None
        self.battery: Battery = None
        self.M: float = None

        self.model: Model = None

    def env(self, architecture: BasicMicroGridArchitecture, horizon: int):
        self.horizon = horizon
        self.pv_system = architecture.pv_system
        self.battery = architecture.battery
        self.M = self.battery.nominal_capacity

        self.model = self._build()

    def _build(self) -> Model:
        opt_model = AbstractModel()

        opt_model.hour = RangeSet(0, self.horizon - 1)
        opt_model.battery_outflow = Set()
        opt_model.pv_panel_outflow = Set()
        opt_model.grid_inflow = Set()

        opt_model.pv_generation = Param(opt_model.hour, domain=NonNegativeReals)
        opt_model.energy_price = Param(opt_model.hour, domain=NonNegativeReals)
        opt_model.loads = Param(opt_model.hour, domain=NonNegativeReals)
        opt_model.export_discount_factor = Param(domain=NonNegativeReals)

        opt_model.y_charge = Var(opt_model.hour, domain=Boolean, initialize=0)
        opt_model.y_discharge = Var(opt_model.hour, domain=Boolean, initialize=0)
        opt_model.grid_to = Var(opt_model.hour, opt_model.grid_inflow, within=NonNegativeReals, initialize=0)
        opt_model.pv_to = Var(opt_model.hour, opt_model.pv_panel_outflow, within=NonNegativeReals, initialize=0)
        opt_model.battery_to = Var(opt_model.hour, opt_model.battery_outflow, within=NonNegativeReals, initialize=0)

        def obj_function(model):
            return sum(model.export_discount_factor * model.energy_price[t] * (
                    model.battery_to[t, 'grid'] * self.battery.discharging_efficiency + model.pv_to[t, 'grid']) for t in
                       model.hour) - sum(
                model.energy_price[t] * (model.grid_to[t, 'household'] + model.grid_to[t, 'battery']) for t in
                model.hour) - sum(self.battery.costs_of_operation * (
                    model.pv_to[t, 'battery'] + model.grid_to[t, 'battery'] + model.battery_to[t, 'household'] +
                    model.battery_to[t, 'grid']) for t in model.hour)

        def load_balance_constraint(model, t):
            return model.pv_to[t, 'household'] + model.battery_to[
                t, 'household'] * self.battery.discharging_efficiency + model.grid_to[t, 'household'] == \
                model.loads[t]

        def pv_generation_constraint(model, t):
            return sum(model.pv_to[t, i] for i in model.pv_panel_outflow) == model.pv_generation[
                t] * self.pv_system.capacity

        def max_charging_constraint_0(model, t):
            return model.pv_to[t, 'battery'] + model.grid_to[t, 'battery'] <= self.battery.max_charging_rate * \
                model.y_charge[t]

        def max_charging_constraint_1(model, t):
            current_state_of_charge = self._current_state_of_charge(model=model, t=t)
            return model.pv_to[t, 'battery'] + model.grid_to[t, 'battery'] <= (
                    self.battery.max_status_of_charge - current_state_of_charge) * self.battery.nominal_capacity

        def max_discharging_constraint_0(model, t):
            return model.battery_to[t, 'household'] + model.battery_to[
                t, 'grid'] <= self.battery.max_discharging_rate * model.y_discharge[t]

        def max_discharging_constraint_1(model, t):
            current_state_of_charge = self._current_state_of_charge(model=model, t=t)
            return model.battery_to[t, 'household'] + model.battery_to[
                t, 'grid'] <= (
                    current_state_of_charge - self.battery.min_status_of_charge) * self.battery.nominal_capacity

        def simul_battery_action(model, t):
            return model.y_charge[t] + model.y_discharge[t] <= 1

        # Objective function
        opt_model.obj = Objective(sense=maximize, rule=obj_function)

        # Constraints
        opt_model.load_balance_constraint = Constraint(opt_model.hour, rule=load_balance_constraint)
        opt_model.pv_generation_constraint = Constraint(opt_model.hour, rule=pv_generation_constraint)
        opt_model.max_charging_constraint_0 = Constraint(opt_model.hour, rule=max_charging_constraint_0)
        opt_model.max_charging_constraint_1 = Constraint(opt_model.hour, rule=max_charging_constraint_1)
        opt_model.max_discharging_constraint_0 = Constraint(opt_model.hour, rule=max_discharging_constraint_0)
        opt_model.max_discharging_constraint_1 = Constraint(opt_model.hour, rule=max_discharging_constraint_1)
        opt_model.simul_battery_action = Constraint(opt_model.hour, rule=simul_battery_action)

        return opt_model

    def _current_state_of_charge(self, model, t):
        e_in = (sum(model.pv_to[i, 'battery'] for i in range(0, t)) + sum(
            model.grid_to[i, 'battery'] for i in range(0, t))) * self.battery.charging_efficiency
        e_out = sum(model.battery_to[i, 'household'] for i in range(0, t)) + sum(
            model.battery_to[i, 'grid'] for i in range(0, t))
        return self.battery.initial_status_of_charge + ((e_in - e_out) / self.battery.nominal_capacity)

    def solve(self, data: Dict):
        instance = self.model.create_instance(data={None: data})

        # Solve the optimization problem
        solver = SolverFactory('glpk', executable=PATH_TO_GLPK_EXECUTABLE)
        results = solver.solve(instance)
        instance.solutions.store_to(results)
        return instance

    def log_results(self, instance, architecture: BasicMicroGridArchitecture, horizon: int) -> pd.DataFrame:
        results = pd.DataFrame()
        for t in range(horizon):
            e_in = sum(value(instance.pv_to[i, 'battery']) for i in range(0, t)) + sum(
                value(instance.grid_to[i, 'battery']) for i in range(0, t))
            e_out = sum(value(instance.battery_to[i, 'household']) for i in range(0, t)) + sum(
                value(instance.battery_to[i, 'grid']) for i in range(0, t))

            current_status_of_charge = architecture.battery.initial_status_of_charge + (
                    (e_in - e_out) / architecture.battery.nominal_capacity)
            info = {
                "load": architecture.household.data()[t],
                "power_generated": architecture.pv_system.data()[t],
                "energy_price": architecture.utility_grid.data()[t],
                "soc_battery": current_status_of_charge,
                "grid_action": [value(instance.grid_to[t, 'household']), value(instance.grid_to[t, 'battery'])],
                "pv_action": [value(instance.pv_to[t, 'household']), value(instance.pv_to[t, 'battery']),
                              value(instance.pv_to[t, 'grid'])],
                "battery_action": [value(instance.battery_to[t, 'household']), value(instance.battery_to[t, 'grid'])],
                "reward": (architecture.utility_grid.export_price_discount * value(
                    instance.energy_price[t]) * (
                                   value(instance.battery_to[t, 'grid']) + value(instance.pv_to[t, 'grid']))) - (
                                  value(instance.energy_price[t]) * value(instance.grid_to[t, 'household']) + value(
                              instance.grid_to[t, 'battery'])) - (architecture.battery.costs_of_operation * (
                        value(instance.pv_to[t, 'battery']) + value(instance.grid_to[t, 'battery']) + value(
                    instance.battery_to[t, 'household']) +
                        value(instance.battery_to[t, 'grid']))),
                "revenue": (architecture.utility_grid.export_price_discount * value(
                    instance.energy_price[t]) * (
                                    value(instance.battery_to[t, 'grid']) + value(instance.pv_to[t, 'grid']))),
                "costs": (value(instance.energy_price[t]) * value(instance.grid_to[t, 'household']) + value(
                    instance.grid_to[t, 'battery'])),
                "costs_battery_operation": (architecture.battery.costs_of_operation * (
                        value(instance.pv_to[t, 'battery']) + value(instance.grid_to[t, 'battery']) + value(
                    instance.battery_to[t, 'household']) +
                        value(instance.battery_to[t, 'grid']))),
                "penalty": 0,
                "remaining_steps": horizon - t - 1,
                "cumulative_reward": sum(
                    architecture.utility_grid.export_price_discount * value(instance.energy_price[i]) * (
                            value(instance.battery_to[i, 'grid']) + value(instance.pv_to[i, 'grid'])) for i in
                    range(t + 1)) - sum(
                    value(instance.energy_price[i]) * value(instance.grid_to[i, 'household']) + value(
                        instance.grid_to[i, 'battery']) for i in range(t + 1)) - sum(
                    architecture.battery.costs_of_operation * (
                            value(instance.pv_to[i, 'battery']) + value(instance.grid_to[i, 'battery']) + value(
                        instance.battery_to[i, 'household']) +
                            value(instance.battery_to[i, 'grid'])) for i in range(t + 1)),
                "nominal_capacity_pv": architecture.pv_system.capacity,
                "nominal_capacity_battery": architecture.battery.nominal_capacity,
                "timestamp": architecture.calendar.data()[0],
            }
            results = results.append(info, ignore_index=True)
        return results
