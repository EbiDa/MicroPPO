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

from typing import Union

from microppo.envs.common.components.component import ControllableMicroGridComponent
from microppo.utils.spaces import NumericSpace


class Battery(ControllableMicroGridComponent):

    def __init__(self, nominal_capacity: float, initial_soc: float, min_soc: float, max_soc: float,
                 charging_efficiency: float, discharging_efficiency: float, max_charging_rate: float,
                 max_discharging_rate: float, costs_of_operation: float, state_space: NumericSpace,
                 action_space: NumericSpace):
        """
        Battery as controllable micro-grid component.

        :param float nominal_capacity: Nominal capacity of the battery.
        :param float initial_soc: Initial state of charge.
        :param float min_soc: Minimum allowed state of charge.
        :param float max_soc: Maximum allowed state of charge.
        :param float charging_efficiency: Charging efficiency.
        :param float discharging_efficiency: Discharging efficiency.
        :param float max_charging_rate: Maximum allowed charging rate.
        :param float max_discharging_rate: Maximum allowed discharging rate.
        :param float costs_of_operation: Costs associated with charging/discharging the battery.
        :param NumericSpace state_space: Space of states related to the battery.
        :param NumericSpace action_space: Space of actions related to the battery.
        """
        super().__init__(component_type="battery", state_space=state_space, action_space=action_space,
                         time_series_data=None)

        self._nominal_capacity = nominal_capacity
        self._initial_state_of_charge = initial_soc
        self._state_of_charge = self._initial_state_of_charge

        self._state_of_charge_min = min_soc
        self._state_of_charge_max = max_soc
        self._charging_efficiency = charging_efficiency
        self._discharging_efficiency = discharging_efficiency
        self._max_charging_rate = max_charging_rate
        self._max_discharging_rate = max_discharging_rate

        self._costs_of_operation = costs_of_operation

    def state(self) -> Union[str, int, float, bool]:
        # Return the battery's current state of charge (SOC).
        return self._state_of_charge

    def step(self, *args, **kwargs) -> Union[str, int, float, bool]:
        # Power inflow
        power_in: float = args[0]['battery_power_in'] * self.charging_efficiency if args[0][
                                                                                        'battery_power_in'] is not None else 0

        # Power outflow
        power_out: float = args[0]['battery_power_out'] if args[0]['battery_power_out'] is not None else 0

        # Update the battery's SOC (based on power in-/outflow)
        update = (-power_out + power_in) / self.nominal_capacity
        self._state_of_charge += update
        return self.state()

    def reset(self) -> Union[str, int, float, bool]:
        # Reset the SOC to its initial value
        self._state_of_charge = self._initial_state_of_charge
        return self.state()

    def charge(self, weight: float, input_power: float = None) -> float:
        """
        Charge the battery with a maximum weight * input_power.

        :param float weight: Share of power assigned for charging (in %).
        :param float input_power: Input power.

        :rtype: float
        :return: Actual charging power.
        """
        # Compute maximum upward power as the minimum between the max. charging rate and the available upward power.
        max_upward_power = min(self._max_charging_rate, self.available_upward_power())

        if input_power is not None:
            charging_power = min(input_power * self.charging_efficiency, max_upward_power)
        else:
            charging_power = max_upward_power

        # Return the actual charging power.
        return charging_power * abs(weight)

    def discharge(self, weight: float) -> float:
        """
        Discharge the battery with a maximum weight * available downward power.

        :param float weight: Share of power assigned for discharging (in %).

        :rtype: float
        :return: Actual discharging power.
        """
        # Compute maximum downward power as the minimum between the max. discharging rate and the available downward power.
        max_downward_power = min(self._max_discharging_rate, self.available_downward_power())
        return max_downward_power * abs(weight)

    def available_upward_power(self) -> float:
        """
        Compute the available upward power based on the battery's current SOC, its SOC limits and its capacity.

        :rtype: float
        :return: Available upward power.
        """
        return max(0.0, (self._state_of_charge_max - self._state_of_charge) * self.nominal_capacity)

    def available_downward_power(self) -> float:
        """
        Compute the available downward power based on the battery's current SOC, its SOC limits and its capacity.

        :rtype: float
        :return: Available downward power.
        """
        return max(0.0, (self._state_of_charge - self._state_of_charge_min) * self.nominal_capacity)

    @property
    def nominal_capacity(self) -> float:
        """
        Return the battery's nominal capacity.

        :rtype: float
        :return: Battery's nominal capacity.
        """
        return self._nominal_capacity

    @property
    def costs_of_operation(self) -> float:
        """
        Return the battery's operational costs.

        :rtype: float
        :return: Battery's cost of operation.
        """
        return self._costs_of_operation

    @property
    def charging_efficiency(self) -> float:
        """
        Return the battery's charging efficiency.

        :rtype: float
        :return: Battery's charging efficiency.
        """
        return self._charging_efficiency

    @property
    def discharging_efficiency(self) -> float:
        """
        Return the battery's discharging efficiency.

        :rtype: float
        :return: Battery's discharging efficiency.
        """
        return self._discharging_efficiency

    @property
    def max_charging_rate(self) -> float:
        """
        Return the battery's maximum allowed charging rate.

        :rtype: float
        :return: Battery's maximum charging rate.
        """
        return self._max_charging_rate

    @property
    def max_discharging_rate(self) -> float:
        """
        Return the battery's maximum allowed discharging rate.

        :rtype: float
        :return: Battery's maximum discharging rate.
        """
        return self._max_discharging_rate

    @property
    def max_status_of_charge(self) -> float:
        """
        Return the battery's maximum allowed state of charge (SOC).

        :rtype: float
        :return: Battery's maximum allowed SOC.
        """
        return self._state_of_charge_max

    @property
    def min_status_of_charge(self) -> float:
        """
        Return the battery's minimum allowed state of charge (SOC).

        :rtype: float
        :return: Battery's minimum allowed SOC.
        """
        return self._state_of_charge_min

    @property
    def initial_status_of_charge(self) -> float:
        """
        Return the battery's initial state of charge (SOC).

        :rtype: float
        :return: Battery's initial SOC.
        """
        return self._initial_state_of_charge
