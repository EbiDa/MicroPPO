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
from gym.envs.registration import register
from typing import Dict, List

import microppo.envs.base
from microppo.utils.spaces import ContinuousNumericSpace, NumericSpace, DiscreteNumericSpace, EmptyNumericSpace

DISCRETE_BASIC_MG = "discrete_basic_microgrid-v0"
CONTINUOUS_BASIC_MG = "continuous_basic_microgrid-v0"
PENALIZED_CONTINUOUS_BASIC_MG = "penalized_continuous_basic_microgrid-v0"

register(id=DISCRETE_BASIC_MG, entry_point='microppo.envs.base:DiscreteDecentralizedMicroGridEnv')
register(id=CONTINUOUS_BASIC_MG, entry_point='microppo.envs.base:ContinuousDecentralizedMicroGridEnv')
register(id=PENALIZED_CONTINUOUS_BASIC_MG,
         entry_point='microppo.envs.base:PenalizedContinuousDecentralizedMicroGridEnv')


def get_default_observation_space_specs() -> Dict[str, ContinuousNumericSpace]:
    return {
        "calendar": ContinuousNumericSpace(lower_bound=[0, 1, 0], upper_bound=[23, 12, 1]),
        "household": ContinuousNumericSpace(lower_bound=[0], upper_bound=[100]),
        "utility_grid": ContinuousNumericSpace(lower_bound=[0], upper_bound=[1]),
        "pv_system": ContinuousNumericSpace(lower_bound=[0], upper_bound=[100]),
        "battery": ContinuousNumericSpace(lower_bound=[0], upper_bound=[1])
    }


def get_default_action_space_specs(env_id: str) -> Dict[str, NumericSpace]:
    if env_id == CONTINUOUS_BASIC_MG:
        return {
            "pv_system": ContinuousNumericSpace(lower_bound=[0, -1], upper_bound=[1, 1]),
            "battery": ContinuousNumericSpace(lower_bound=[-1], upper_bound=[1])
        }
    elif env_id == PENALIZED_CONTINUOUS_BASIC_MG:
        return {
            "pv_system": ContinuousNumericSpace(lower_bound=[0, 0], upper_bound=[1, 1]),
            "battery": ContinuousNumericSpace(lower_bound=[0, 0, 0], upper_bound=[1, 1, 1])
        }
    elif env_id == DISCRETE_BASIC_MG:
        pv_actions = np.hstack((np.arange(0.1, 1.1, 0.1), np.arange(0.1, 1.1, 0.1), np.array(1)))
        battery_actions = np.hstack(
            (np.arange(0.1, 1.1, 0.1), np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), np.array(0)))

        return {
            "pv_system": DiscreteNumericSpace(n_options=len(pv_actions), values=list(pv_actions)),
            "battery": DiscreteNumericSpace(n_options=len(battery_actions), values=list(battery_actions))
        }
    elif env_id is None:
        return {
            "pv_system": EmptyNumericSpace(),
            "battery": EmptyNumericSpace()
        }
    else:
        raise ValueError(env_id)


def default_action_space_masking(actions) -> List[bool]:
    is_allowed_action = [
        False if 10 <= action[0] < 15 and (not 20 <= action[1] <= 30) else True for action in
        actions]
    return is_allowed_action
