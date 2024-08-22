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
import os
import random
import torch

from microppo.envs import PENALIZED_CONTINUOUS_BASIC_MG, DISCRETE_BASIC_MG, CONTINUOUS_BASIC_MG
from microppo.model import MicroPPOPolicy

ENVIRONMENT_MAPPING = {
    "ppo_c": PENALIZED_CONTINUOUS_BASIC_MG,
    "ppo_d": DISCRETE_BASIC_MG,
    "dqn": DISCRETE_BASIC_MG,
    "microppo": CONTINUOUS_BASIC_MG,
    "milp": None,
    "seq_milp": None,
    "rb_economic": None,
    "rb_own": None
}

EXP_SEED = 10

N_SIM_ENVS = 24

MODEL_PARAMETERS = {
    "ppo_c": dict(policy="MlpPolicy", env=None, verbose=0, learning_rate=0.005, n_steps=7, n_epochs=3, batch_size=168,
                  device="cpu", policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[32, 32])), ent_coef=0.1),
    "ppo_d": dict(policy="MlpPolicy", env=None, verbose=0, learning_rate=0.005, n_steps=7, n_epochs=1, batch_size=168,
                  device="cpu", policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[32, 32])), ent_coef=0.1),
    "dqn": dict(policy="MlpPolicy", env=None, verbose=0, learning_rate=0.005, target_update_interval=168,
                learning_starts=24, device="cpu"),
    "microppo": dict(policy=MicroPPOPolicy, env=None, verbose=0, learning_rate=0.0085, n_steps=7, n_epochs=1,
                     batch_size=168, ent_coef=0.1, device="cpu"),
    "rb_economic": dict(policy="economic"),
    "rb_own": dict(policy="own_consumption")
}

DATA_BASE_FOLDER = "data/2019-168"
DATA_FOLDER_SUFFIX_FORECASTS = "-fc"
DATA_FILES_PREFIX = "DE-2019_loads_pv_prices"

PATH_TO_GLPK_EXECUTABLE = "~/miniconda3/envs/<CONDA_ENV_NAME>/bin/glpsol"

PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}% {bar} | {elapsed}<{remaining}"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
