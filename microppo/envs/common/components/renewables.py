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
from typing import Union, List

from microppo.envs.common.components.component import ControllableMicroGridComponent
from microppo.utils.spaces import NumericSpace


class PVSystem(ControllableMicroGridComponent):
    def __init__(self, capacity: float, state_space: NumericSpace,
                 action_space: NumericSpace, time_series_data: Union[np.ndarray, List], start_idx: int = 0):
        """
        Photovoltaic (PV) System as controllable micro-grid component.

        :param float capacity: Nominal capacity of the PV system.
        :param NumericSpace state_space: Space of states related to the PV system.
        :param NumericSpace action_space: Space of actions related to the PV system.
        :param Union[np.ndarray, List] time_series_data: Data on the power generated.
        :param int start_idx: Index which is viewed as initial time point (optional, default is 0).
        """
        super().__init__(component_type="pv_panels", state_space=state_space, action_space=action_space,
                         time_series_data=time_series_data, start_idx=start_idx)
        self._capacity = capacity

    def state(self) -> Union[str, int, float, bool]:
        return self._time_series_data[self._step % len(self._time_series_data)] * self.capacity

    def step(self, *args, **kwargs) -> Union[str, int, float, bool]:
        self._step += 1
        return self.state()

    def reset(self) -> Union[str, int, float, bool]:
        self._step = self._initial_step
        return self.state()

    @property
    def capacity(self) -> float:
        return self._capacity
