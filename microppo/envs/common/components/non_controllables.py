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
from datetime import datetime
from typing import Union, List, Dict, Tuple

from microppo.envs.common.components.component import MicroGridComponent
from microppo.utils.spaces import NumericSpace


class UtilityGrid(MicroGridComponent):
    def __init__(self, export_price_discount: float, state_space: NumericSpace,
                 time_series_data: Union[np.ndarray, List], start_idx: int = 0):
        """
        Connection to the utility grid as non-controllable micro-grid component.
        
        :param float export_price_discount: Discount price factor in [0,1] for selling power to the utility grid.
        :param NumericSpace state_space: Space of states related to the utility grid.
        :param Union[np.ndarray, List] time_series_data: Data on the energy prices.
        :param int start_idx: Index which is viewed as initial time point (optional, default is 0).
        """
        super().__init__(component_type="grid_connection", state_space=state_space, time_series_data=time_series_data,
                         start_idx=start_idx)
        self._export_price_discount = export_price_discount

    @property
    def export_price_discount(self) -> float:
        """
        Return the discount price factor.

        :rtype: float
        :return: Discount price factor for selling power to the utility grid.
        """
        return self._export_price_discount


class Household(MicroGridComponent):
    def __init__(self, state_space: NumericSpace, time_series_data: Union[np.ndarray, List], start_idx: int = 0):
        """
        Household as non-controllable micro-grid component.

        :param NumericSpace state_space: Space of states related to the household.
        :param Union[np.ndarray, List] time_series_data: Data on the local loads.
        :param int start_idx: Index which is viewed as initial time point (optional, default is 0).
        """
        super().__init__(component_type="household_loads", state_space=state_space, time_series_data=time_series_data,
                         start_idx=start_idx)


class Calendar(MicroGridComponent):
    def __init__(self, state_space: NumericSpace, time_series_data: Union[np.ndarray, List], start_idx: int = 0):
        """
        Calendar as non-controllable micro-grid component.

        :param NumericSpace state_space: Space of states related to the household.
        :param Union[np.ndarray, List] time_series_data: Data on the date.
        :param int start_idx: Index which is viewed as initial time point (optional, default is 0).
        """
        super().__init__(component_type="time_indication", state_space=state_space, time_series_data=time_series_data,
                         start_idx=start_idx)

    def state(self) -> List[int]:
        """
        Return the hour and month of the date as well as whether it is a working day or not.

        :rtype: List[int]
        :return: Current state of the calendar.
        """
        month, weekday, hour = self._get_information_from_timestamp()
        state: Dict[str, int] = {'hour': hour, 'weekday': 1 if weekday < 5 else 0, 'month': month}
        return list(state.values())

    def _get_information_from_timestamp(self) -> Tuple[int, int, int]:
        """
        Extract month, weekday and hour from  current timestamp.

        :rtype: Tuple[int, int, int]
        :return: Month, weekday and hour.
        """
        timestamp = datetime.strptime(self._time_series_data[self._step % len(self._time_series_data)],
                                      '%Y-%m-%d %H:%M:%S')
        month = timestamp.month
        weekday = timestamp.weekday()
        hour = timestamp.hour
        return month, weekday, hour
