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
import json
import numpy as np
import os
import pandas as pd
from abc import abstractmethod
from datetime import datetime
from gym import Env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from typing import Any, List, Dict, Union, Optional, Tuple, Callable

from experiments.baselines.milp import MicroGridMILP
from experiments.baselines.rule_based_approach import RuleBasedApproach
from experiments.baselines.seq_milp import SequentialMicroGridMILP
from experiments.config import MODEL_PARAMETERS, ENVIRONMENT_MAPPING, N_SIM_ENVS
from microppo.envs import get_default_observation_space_specs, get_default_action_space_specs, \
    default_action_space_masking, DISCRETE_BASIC_MG
from microppo.envs.common.architectures import BasicMicroGridArchitecture
from microppo.envs.common.components import Calendar, Household, UtilityGrid, PVSystem, Battery
from microppo.model import MicroPPO
from microppo.utils.spaces import NumericSpace


def make_env(env_id: str, architecture: BasicMicroGridArchitecture, horizon: int) -> gym.Env:
    def _init() -> gym.Env:

        if env_id == DISCRETE_BASIC_MG:
            env = gym.make(env_id, architecture=architecture, horizon=horizon,
                           action_space_masking=default_action_space_masking)
        else:
            env = gym.make(env_id, architecture=architecture, horizon=horizon)
        env.reset()
        return env

    return _init()


def make_vectorized_env(env_id: str, architecture: BasicMicroGridArchitecture, horizon: int, n_envs: int = 1) -> VecEnv:
    if env_id == DISCRETE_BASIC_MG:
        return make_vec_env(env_id, n_envs=n_envs, env_kwargs={'architecture': architecture, 'horizon': horizon,
                                                               "action_space_masking": default_action_space_masking})
    else:
        return make_vec_env(env_id, n_envs=n_envs, env_kwargs={'architecture': architecture, 'horizon': horizon})


def _build_basic_microgrid_architecture(calendar_params: Dict[str, Union[np.ndarray, List, NumericSpace]],
                                        pv_system_params: Dict[str, Union[np.ndarray, List, float, NumericSpace]],
                                        battery_params: Dict[str, Union[np.ndarray, List, float, NumericSpace]],
                                        load_params: Dict[str, Union[np.ndarray, List, float, NumericSpace]],
                                        grid_params: Dict[str, Union[
                                            np.ndarray, List, float, NumericSpace]]) -> BasicMicroGridArchitecture:
    calendar = Calendar(state_space=calendar_params['state_space'], time_series_data=calendar_params['time_series'])

    household = Household(state_space=load_params['state_space'], time_series_data=load_params['time_series'])

    utility_grid = UtilityGrid(export_price_discount=grid_params['export_price_discount'],
                               state_space=grid_params['state_space'], time_series_data=grid_params['time_series'])

    pv_system = PVSystem(capacity=pv_system_params['capacity'], state_space=pv_system_params['state_space'],
                         action_space=pv_system_params['action_space'],
                         time_series_data=pv_system_params['time_series'])

    battery = Battery(nominal_capacity=battery_params['nominal_capacity'], initial_soc=battery_params['initial_soc'],
                      min_soc=battery_params['min_soc'], max_soc=battery_params['max_soc'],
                      charging_efficiency=battery_params['charging_efficiency'],
                      discharging_efficiency=battery_params['discharging_efficiency'],
                      max_charging_rate=battery_params['max_charging_rate'],
                      max_discharging_rate=battery_params['max_discharging_rate'],
                      costs_of_operation=battery_params['costs_of_operation'],
                      state_space=battery_params['state_space'], action_space=battery_params['action_space'])

    return BasicMicroGridArchitecture(calendar=calendar, household=household, utility_grid=utility_grid,
                                      pv_system=pv_system, battery=battery)


def _create_environment(model_id: str, input_file_path: str, make_vec_env: Optional[bool] = False,
                        n_envs: Optional[int] = 1) -> Union[Tuple[
    Union[VecEnv, Env, BasicMicroGridArchitecture], int], Union[Env]]:
    df = pd.read_csv(input_file_path, sep=',', header=0)
    timestamps = np.array(df['timestamp'], dtype=str)

    load_data = np.array(df['load'], dtype=float)
    pv_system_factor = np.array(df['pv_capacity_factor'], dtype=float)
    energy_prices = np.array(df['energy_price'], dtype=float)
    horizon = len(df)

    json_file_path = os.path.join(input_file_path.rsplit('/', 1)[0],
                                  "DE-2019_info_" + input_file_path.rsplit('/', 1)[0].rsplit("/", 1)[1] + ".json")
    with open(json_file_path, 'r') as json_file:
        json_conf = json.load(json_file)

    annual_consumption = json_conf["annual_consumption"]
    pv_capacity = round(annual_consumption / 1000 * 1.5, 1)
    battery_capacity = round(annual_consumption / 1000, 1)

    if battery_capacity <= 6:
        battery_cost = battery_capacity * 1100
    elif battery_capacity <= 9:
        battery_cost = battery_capacity * 800
    else:
        battery_cost = battery_capacity * 600
    battery_operation_cost = round((battery_cost / (battery_capacity * 0.8 * 10000 * 0.9)), 4)

    calendar_params = {
        "state_space": get_default_observation_space_specs()['calendar'],
        "time_series": timestamps}
    household_params = {
        "state_space": get_default_observation_space_specs()['household'],
        "time_series": load_data}
    grid_params = {
        "export_price_discount": 0.25,
        "state_space": get_default_observation_space_specs()['utility_grid'],
        "time_series": energy_prices}
    pv_system_params = {
        "capacity": pv_capacity,
        "state_space": get_default_observation_space_specs()['pv_system'],
        "action_space": get_default_action_space_specs(env_id=ENVIRONMENT_MAPPING[model_id])['pv_system'],
        "time_series": pv_system_factor}
    battery_params = {
        "nominal_capacity": battery_capacity, "initial_soc": 0.5, "min_soc": 0.1, "max_soc": 0.9,
        "charging_efficiency": 0.9, "discharging_efficiency": 0.9,
        "max_charging_rate": 0.5 * battery_capacity,
        "max_discharging_rate": 0.5 * battery_capacity, "costs_of_operation": battery_operation_cost,
        "state_space": get_default_observation_space_specs()['battery'],
        "action_space": get_default_action_space_specs(env_id=ENVIRONMENT_MAPPING[model_id])['battery']}

    architecture = _build_basic_microgrid_architecture(calendar_params=calendar_params,
                                                       load_params=household_params,
                                                       grid_params=grid_params, pv_system_params=pv_system_params,
                                                       battery_params=battery_params)

    if ENVIRONMENT_MAPPING[model_id] is None:
        return architecture, horizon
    else:
        if make_vec_env:
            env = make_vectorized_env(ENVIRONMENT_MAPPING[model_id], architecture=architecture, horizon=horizon,
                                      n_envs=n_envs)
        else:
            env = make_env(ENVIRONMENT_MAPPING[model_id], architecture=architecture, horizon=horizon)
        return env, horizon


def _create_single_environment(model_id: str, input_file_path: str) -> Callable:
    env, _ = _create_environment(model_id, input_file_path)
    return lambda: env


class DummyModel:

    def __init__(self, model_id: str, algorithm: Any, is_savable: bool = False):
        self.id = model_id
        self._algorithm = algorithm
        self._is_savable = is_savable

        # experiment run id
        self.run_id = None

    def train(self, datasets: List[str], fold_id: int = 1):
        self._train(datasets=datasets, fold_id=fold_id)

        if self._is_savable:
            self._save(fold_id=fold_id)

        return self

    def evaluate(self, datasets: List[str], fold_id: int = 1) -> pd.DataFrame:
        return self._evaluate(datasets=datasets, fold_id=fold_id)

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    @abstractmethod
    def _train(self, datasets: List[str], fold_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self, datasets: List[str], fold_id: int) -> pd.DataFrame:
        raise NotImplementedError

    def _save(self, fold_id: int) -> None:
        date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not self.run_id:
            output_path = "output/models/" + date + "/__unassigned__/" + self.id
            suffix = str(fold_id)
        else:
            output_path = "output/models/" + date + "/" + self.run_id + "/" + self.id
            suffix = self.run_id + "_" + str(fold_id)

        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        except FileExistsError:
            print("Output folder already exists.")

        model_path = output_path + "/" + self.id + "_" + timestamp + "_" + suffix
        self._algorithm.save(model_path)

        if self.id in list(MODEL_PARAMETERS.keys()):
            json_path = output_path + "/" + self.id + "_" + date + "_" + self.run_id + '_params.json'
            if not os.path.exists(json_path):
                with open(json_path, 'w') as file:
                    params = MODEL_PARAMETERS[self.id]
                    params.pop('env', None)
                    if "policy" in list(params.keys()):
                        params["policy"] = str(type(params["policy"]))
                    json.dump(params, file)


class DummyDRLModel(DummyModel):
    def __init__(self, model_id: str, algorithm: Any, is_savable: bool = False):
        super().__init__(model_id=model_id, algorithm=algorithm, is_savable=is_savable)

    def _train(self, datasets: List[str], fold_id: int) -> None:
        if len(datasets) > 0:
            _, horizon = _create_environment(model_id=self.id, input_file_path=datasets[0])

            for i in range(0, len(datasets) - N_SIM_ENVS + 1, N_SIM_ENVS):
                partition = datasets[i:i + N_SIM_ENVS]
                env_list = [_create_single_environment(model_id=self.id, input_file_path=file_path) for file_path in
                            partition]
                vec_env = DummyVecEnv(env_list)

                self._algorithm.env = vec_env
                self._algorithm.env.reset()
                self._algorithm.learn(total_timesteps=horizon, progress_bar=False)

    def _evaluate(self, datasets: List[str], fold_id: int) -> pd.DataFrame:
        records = pd.DataFrame()

        if len(datasets) > 0:
            for file_path in datasets:
                env, horizon = _create_environment(model_id=self.id, input_file_path=file_path, make_vec_env=True)
                self._algorithm.env = env

                vec_env = self._algorithm.get_env()
                obs = vec_env.reset()

                for step in range(horizon):
                    action, _state = self._algorithm.predict(observation=obs, deterministic=True)
                    obs, reward, done, info = vec_env.step(actions=action)

                    log = info.pop()
                    log['timestamp'] = vec_env.get_attr("architecture", 0)[0].calendar.data()[0]
                    log['fold'] = fold_id
                    records = records.append(other=log, ignore_index=True)
        return records


class DummyMILPModel(DummyModel):

    def __init__(self, model_id: str, algorithm: Any, is_savable: bool = False):
        super().__init__(model_id=model_id, algorithm=algorithm, is_savable=is_savable)

    def _train(self, datasets: List[str], fold_id: int) -> None:
        pass

    def _evaluate(self, datasets: List[str], fold_id: int) -> pd.DataFrame:
        records = pd.DataFrame()

        if len(datasets) > 0:
            for file_path in datasets:
                architecture, horizon = _create_environment(model_id=self.id, input_file_path=file_path)
                self._algorithm.env(architecture=architecture, horizon=horizon)

                data = {
                    'hour': {None: (t for t in range(0, horizon))},
                    'battery_outflow': {None: ('household', 'grid')},
                    'pv_panel_outflow': {None: ('household', 'battery', 'grid')},
                    'grid_inflow': {None: ('household', 'battery')},
                    'pv_generation': {t: obs for _, (obs, t) in
                                      enumerate(zip(architecture.pv_system.data(), range(0, horizon)))},
                    'energy_price': {t: obs for _, (obs, t) in
                                     enumerate(zip(architecture.utility_grid.data(), range(0, horizon)))},
                    'loads': {t: obs for _, (obs, t) in
                              enumerate(zip(architecture.household.data(), range(0, horizon)))},
                    'export_discount_factor': {None: architecture.utility_grid.export_price_discount}
                }

                solved_instance = self._algorithm.solve(data=data)
                log = self._algorithm.log_results(instance=solved_instance, architecture=architecture, horizon=horizon)
                if fold_id is not None:
                    log['fold'] = fold_id
                records = records.append(other=log, ignore_index=True)
        return records


class DummySequentialMILPModel(DummyModel):

    def __init__(self, model_id: str, algorithm: Any, is_savable: bool = False):
        super().__init__(model_id=model_id, algorithm=algorithm, is_savable=is_savable)

    def _train(self, datasets: List[str], fold_id: int) -> None:
        pass

    def _evaluate(self, datasets: List[str], fold_id: int) -> pd.DataFrame:
        records = pd.DataFrame()

        if len(datasets) > 0:
            for file_path in datasets:
                architecture, horizon = _create_environment(model_id=self.id, input_file_path=file_path)
                self._algorithm.env(architecture=architecture, horizon=horizon)
                for step in range(horizon):
                    action = self._algorithm.solve()
                    obs, reward, done, info = self._algorithm.step(action=action)

                    log = info.pop()
                    log['timestamp'] = architecture.calendar.data()[0]
                    log['fold'] = fold_id
                    records = records.append(other=log, ignore_index=True)
        return records


class DummyRuleBasedModel(DummyModel):

    def __init__(self, model_id: str, algorithm: Any, is_savable: bool = False):
        super().__init__(model_id=model_id, algorithm=algorithm, is_savable=is_savable)

    def _train(self, datasets: List[str], fold_id: int) -> None:
        pass

    def _evaluate(self, datasets: List[str], fold_id: int) -> pd.DataFrame:
        records = pd.DataFrame()

        if len(datasets) > 0:
            for file_path in datasets:
                architecture, horizon = _create_environment(model_id=self.id, input_file_path=file_path)
                self._algorithm.env(architecture=architecture, horizon=horizon)
                for step in range(horizon):
                    action = self._algorithm.apply_rules()
                    obs, reward, done, info = self._algorithm.step(action=action)

                    log = info.pop()
                    log['timestamp'] = architecture.calendar.data()[0]
                    log['fold'] = fold_id
                    records = records.append(other=log, ignore_index=True)
        return records


class ModelFactory:
    def __init__(self):
        pass

    def create(self, model_id: str, data_sample: Optional[str] = None) -> DummyModel:
        if model_id == "ppo_c":
            envs, _ = _create_environment(model_id=model_id, input_file_path=data_sample, make_vec_env=True,
                                          n_envs=N_SIM_ENVS)
            model_params = MODEL_PARAMETERS[model_id]
            model_params["env"] = envs
            return DummyDRLModel(model_id=model_id, algorithm=PPO(**model_params), is_savable=True)
        elif model_id == "ppo_d":
            envs, _ = _create_environment(model_id=model_id, input_file_path=data_sample, make_vec_env=True,
                                          n_envs=N_SIM_ENVS)
            model_params = MODEL_PARAMETERS[model_id]
            model_params["env"] = envs
            return DummyDRLModel(model_id=model_id, algorithm=PPO(**model_params), is_savable=True)
        elif model_id == "dqn":
            envs, _ = _create_environment(model_id=model_id, input_file_path=data_sample, make_vec_env=True,
                                          n_envs=N_SIM_ENVS)
            model_params = MODEL_PARAMETERS[model_id]
            model_params["env"] = envs
            return DummyDRLModel(model_id=model_id, algorithm=DQN(**model_params), is_savable=True)
        elif model_id == "microppo":
            envs, _ = _create_environment(model_id=model_id, input_file_path=data_sample, make_vec_env=True,
                                          n_envs=N_SIM_ENVS)
            model_params = MODEL_PARAMETERS[model_id]
            model_params["env"] = envs
            return DummyDRLModel(model_id=model_id, algorithm=MicroPPO(**model_params), is_savable=True)
        elif model_id == "milp":
            return DummyMILPModel(model_id=model_id, algorithm=MicroGridMILP(), is_savable=False)
        elif model_id == "seq_milp":
            return DummySequentialMILPModel(model_id=model_id, algorithm=SequentialMicroGridMILP(), is_savable=False)
        elif model_id == "rb_economic":
            return DummyRuleBasedModel(model_id=model_id, algorithm=RuleBasedApproach(**MODEL_PARAMETERS[model_id]),
                                       is_savable=False)
        elif model_id == "rb_own":
            return DummyRuleBasedModel(model_id=model_id, algorithm=RuleBasedApproach(**MODEL_PARAMETERS[model_id]),
                                       is_savable=False)
        else:
            raise ValueError(model_id)
