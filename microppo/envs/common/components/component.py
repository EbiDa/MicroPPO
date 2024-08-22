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
from abc import ABC, abstractmethod
from typing import Union, List

from microppo.utils.spaces import NumericSpace


class MicroGridComponent(ABC):

    def __init__(self, component_type: str, state_space: NumericSpace, time_series_data: Union[np.ndarray, List],
                 start_idx: int = 0):
        """
        Data-driven micro-grid component.

        :param str component_type: Type of micro-grid component.
        :param NumericSpace state_space: Corresponding state space.
        :param Union[np.ndarray, List] time_series_data: Data that describes the state of the component over time.
        :param int start_idx: Index which is viewed as initial time point (optional, default is zero).
         """
        self._component_type = component_type
        self._state_space = state_space

        self._time_series_data = time_series_data

        self._initial_step = start_idx
        self._step = self._initial_step

    @property
    def component_type(self) -> str:
        """
        Return the type of the micro-grid component.

        :rtype: str
        :return: Type of the micro-grid component.
        """
        return self._component_type

    @property
    def state_space(self) -> NumericSpace:
        """
        Return the numeric space that includes all possible states of the micro-grid component.

        :rtype: NumericSpace
        :return: State space.
        """
        return self._state_space

    @state_space.setter
    def state_space(self, space: NumericSpace):
        self._state_space = space

    def state(self) -> Union[str, int, float, bool]:
        """
        Return the current state of the micro-grid component.

        :rtype: Union[str, int, float, bool]
        :return: Current state of the micro-grid component.
        """
        return self._time_series_data[self._step % len(self._time_series_data)]

    def step(self) -> Union[str, int, float, bool]:
        """
        Perform an update step w.r.t. the micro-grid component and returns its next state.

        :param args: Non-keyworded variable length of arguments.
        :param kwargs: Keyworded variable length of arguments.

        :rtype: Union[str, int, float, bool]
        :return: Next state of the micro-grid component.
        """
        self._step += 1
        return self.state()

    def reset(self) -> Union[str, int, float, bool]:
        """
        Reset the micro-grid component to its initial state.

        :rtype: Union[str, int, float, bool]
        :return: State after reset.
        """
        self._step = self._initial_step
        return self.state()

    def update_data(self, time_series_data: List[Union[float, int]]):
        """
        Update the time series data that contains the state information regarding the micro-grid component.

        :param List[Union[float, int]] time_series_data: Time series data that contains the new state information.
        """
        self._time_series_data = time_series_data
        self.reset()

    def data(self) -> List[Union[float, int]]:
        """
        Return the whole time series data that contains the state information regarding the micro-grid component.

        :rtype:  List[Union[float, int]]
        :return: Time series data that contains the state information.
        """
        return self._time_series_data


class ControllableMicroGridComponent(MicroGridComponent):

    def __init__(self, component_type: str, state_space: NumericSpace, action_space: NumericSpace,
                 time_series_data: Union[np.ndarray, List] = None, start_idx: int = 0):
        """
        Controllable micro-grid component.

        :param str component_type: Type of micro-grid component.
        :param NumericSpace state_space: Corresponding state space.
        :param NumericSpace action_space: Corresponding action space.
        :param Union[np.ndarray, List] time_series_data: Data that describes the state of the component over time (optional, default is None).
        :param int start_idx: Index which is viewed as initial time point (optional, default is 0).
         """
        super().__init__(component_type=component_type, state_space=state_space, time_series_data=time_series_data,
                         start_idx=start_idx)
        self._action_space = action_space

    @property
    def action_space(self) -> NumericSpace:
        """
        Return the numeric space that includes all possible actions w.r.t. the micro-grid component.

        :rtype: Numeric Space
        :return: Action space.
        """
        return self._action_space

    @action_space.setter
    def action_space(self, space: NumericSpace):
        self._action_space = space

    def state(self) -> Union[str, int, float, bool]:
        """
        Return the current state of the micro-grid component.

        :rtype: Union[str, int, float, bool]
        :return: current state
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, **kwargs) -> Union[str, int, float, bool]:
        """
        Perform an update step w.r.t. the micro-grid component and returns its next state.

        :param args: Non-keyworded variable length of arguments.
        :param kwargs: Keyworded variable length of arguments.

        :rtype:
        :return: Next state.
               """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Union[str, int, float, bool]:
        """
        Reset the micro-grid component to its initial state.

        :rtype: Union[str, int, float, bool]
        :return: State after reset.
        """
        raise NotImplementedError
