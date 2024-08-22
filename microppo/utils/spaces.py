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
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List


class NumericSpace(ABC):

    def __init__(self):
        """
        Abstract numeric space.
        """
        pass

    @abstractmethod
    def create(self) -> gym.spaces.Space:
        """
        Create a gym.spaces.Space from numeric space.

        :rtype: gym.spaces.Space
        :return: gym.spaces.Space
        """
        raise NotImplementedError


class DiscreteNumericSpace(NumericSpace):
    def __init__(self, n_options: int, values: List[Union[int, float]] = None):
        """
        Discrete numeric space.

        :param int n_options: Number of elements (i.e., options) within the space.
        :param List[Union[int, float]] values: Corresponding numeric values.
        """
        super().__init__()
        self._n_options = n_options
        self._values = values

    @property
    def n_options(self) -> int:
        """
        Return the number of elements within the space.

        :rtype: int
        :return: Number of options within the space.
        """
        return self._n_options

    @property
    def values(self) -> List[Union[int, float]]:
        """
        Return the elements (i.e., numeric values) within the space.

        :rtype: List[Union[int, float]]
        :return: List containing all elements within the space.
        """
        return self._values

    def value(self, option: int) -> Union[float, int, str, bool]:
        """
        Return numeric value of a certain option.

        :param int option: Index of the option of interest.

        :type: Union[float, int, str, bool]
        :return: Value that corresponds to the option specified by the given index.
        """
        if self.values is not None:
            return self.values[option]
        else:
            return option

    def create(self) -> gym.spaces.Space:
        """
        Create a discrete gym.spaces.Space from the given numeric space.

        :rtype: gym.spaces.Discrete
        :return: Discrete gym.spaces.Space (gym.spaces.Discrete).
        """
        return gym.spaces.Discrete(n=self.n_options)


class ContinuousNumericSpace(NumericSpace):
    def __init__(self, lower_bound: List[Union[int, float]], upper_bound: List[Union[int, float]]):
        """
        Continuous numeric space.

        :param List[Union[int, float]] lower_bound: List containing the lower bounds in all dimensions of the space.
        :param List[Union[int, float]] upper_bound: List containing the upper bounds in all dimensions of the space.
        """
        super().__init__()
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @property
    def lower_bound(self) -> List[Union[int, float]]:
        """
        Return the lower bound for all dimensions of the space.

        :rtype: List[Union[int, float]]
        :return: List containing the lower bounds of the space.
        """
        return self._lower_bound

    @property
    def upper_bound(self) -> List[Union[int, float]]:
        """
        Return the upper bound for all dimensions of the space.

        :rtype: List[Union[int, float]]
        :return: List containing the upper bounds of the space.
        """
        return self._upper_bound

    def create(self) -> gym.spaces.Space:
        """
        Create a continuous gym.spaces.Space from the given numeric space.

        :rtype: gym.spaces.Box
        :return: Continuous gym.spaces.Space (gym.spaces.Box)
        """
        return gym.spaces.Box(low=np.float32(self.lower_bound), high=np.float32(self.upper_bound), dtype=np.float32)


class EmptyNumericSpace(NumericSpace):
    def __init__(self):
        """
        Numeric space that contains no elements.
        """
        super().__init__()

    def create(self) -> gym.spaces.Space:
        """
        Create a discrete gym.spaces.Space that contains zero elements.

        :rtype: gym.spaces.Discrete
        :return: Discrete gym.spaces.Space (gym.spaces.Discrete)
        """
        return gym.spaces.Discrete(n=0)
