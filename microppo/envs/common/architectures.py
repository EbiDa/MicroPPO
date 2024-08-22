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

import itertools
import numpy as np
from dataclasses import dataclass
from typing import List, Union

from microppo.envs.common.components import Calendar, Household, UtilityGrid, PVSystem, Battery
from microppo.envs.common.components.component import MicroGridComponent, ControllableMicroGridComponent


@dataclass
class BasicMicroGridArchitecture:
    """
    Architecture of a basic micro-grid comprising
        - a calendar for time indication
        - local loads of a household
        - a PV system
        - a battery
    """
    calendar: Calendar
    household: Household
    utility_grid: UtilityGrid
    pv_system: PVSystem
    battery: Battery

    def non_controllable_components(self) -> List[MicroGridComponent]:
        """
        Return all non-controllable components of the micro-grid.

        :rtype: List[MicroGridComponent]
        :return: List containing all non-controllable components.
        """
        return [self.calendar, self.household, self.utility_grid]

    def controllable_components(self) -> List[ControllableMicroGridComponent]:
        """
        Return all controllable components of the micro-grid.

        :rtype: List[ControllableMicroGridComponent]
        :return: List containing all controllable components.
        """
        return [self.pv_system, self.battery]

    def all_components(self) -> List[MicroGridComponent]:
        """
        Return all components of the micro-grid.

        :rtype: List[MicroGridComponent]
        :return: List containing all components.
        """
        return self.non_controllable_components() + self.controllable_components()

    def step(self, *args, **kwargs) -> List[Union[str, int, float, bool]]:
        """
        Update the state of the whole micro-grid (i.e., for all micro-grid components) given the selected action

        :param args: Non-keyworded variable length of arguments.
        :param kwargs: Keyworded variable length of arguments.

        :rtype: List[Union[str, int, float, bool]]
        :return: List containing the new state of all  micro-grid components.
        """
        next_states = [component.step() if type(component.state()) == list else [np.array(component.step()).tolist()]
                       for
                       component in self.non_controllable_components()] + [
                          component.step(kwargs) if type(
                              component.state()) == list else [
                              np.array(component.step(kwargs)).tolist()] for
                          component in self.controllable_components()]
        return list(itertools.chain(*next_states))

    def state(self) -> List[Union[str, int, float, bool]]:
        """
        Obtain the current state of the whole micro-grid (i.e., for all micro-grid components)

        :rtype: List[Union[str, int, float, bool]]
        :return: List containing the current state of all micro-grid components.
        """
        states = [
            component.state() if type(component.state()) == list else [np.array(component.state()).tolist()]
            for component in self.all_components()]
        return list(itertools.chain(*states))

    def reset(self) -> List[Union[str, int, float, bool]]:
        initial_states = [
            component.reset() if type(component.state()) == list else [np.array(component.reset()).tolist()]
            for component in self.all_components()]
        return list(itertools.chain(*initial_states))
