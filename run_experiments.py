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

import functools
import itertools
import multiprocessing
import numpy as np
import os
import random
import uuid
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing import List

from experiments.config import DATA_BASE_FOLDER, DATA_FOLDER_SUFFIX_FORECASTS, DATA_FILES_PREFIX, PROGRESS_BAR_FORMAT, \
    EXP_SEED, set_seed
from experiments.experiment import Experiment

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')


def get_data_folds(data_dir: str, prefix: str):
    n_households = 200
    cv_households = KFold(n_splits=10, shuffle=True, random_state=10)

    n_weeks = 52
    holdout_set_weeks = np.array([3, 7, 11, 16, 20, 24, 29, 33, 37, 42, 46, 51])
    train_weeks = [item for item in np.arange(n_weeks) if item not in holdout_set_weeks]

    cnt_fold_households = 0
    folds = []
    for train_households, test_households in cv_households.split(X=np.arange(n_households), y=None):
        train_households += 1
        test_households += 1

        cnt_fold_households += 1
        train_set = list(itertools.chain(*[
            [os.path.join(data_dir, str(d).zfill(3), prefix + '_' + str(d).zfill(3) + '_' + str(week) + '.csv') for
             week in train_weeks] for d in train_households]))
        test_set = list(itertools.chain(*[
            [os.path.join(data_dir, str(d).zfill(3), prefix + '_' + str(d).zfill(3) + '_' + str(week) + '.csv') for
             week in holdout_set_weeks] for d in test_households]))

        set_seed(EXP_SEED)
        random.shuffle(train_set)

        folds.append((cnt_fold_households, train_set, test_set))
    return folds


def run(experiment: Experiment):
    return experiment.run()


def run_experiments(experiments: List[Experiment], n_processes: int = None):
    with multiprocessing.Pool(processes=n_processes) as process_pool:
        res = list(tqdm(process_pool.imap(functools.partial(run), experiments), total=len(experiments),
                        desc="Running experiments", bar_format=PROGRESS_BAR_FORMAT))


if __name__ == "__main__":
    set_seed(EXP_SEED)
    baseline_folds = get_data_folds(data_dir=DATA_BASE_FOLDER, prefix=DATA_FILES_PREFIX)
    forecasted_folds = get_data_folds(data_dir=DATA_BASE_FOLDER + DATA_FOLDER_SUFFIX_FORECASTS,
                                      prefix=DATA_FILES_PREFIX)

    baseline_algorithms = ['milp', 'seq_milp']
    other_algorithms = ['microppo', 'ppo_c', 'ppo_d', 'dqn', 'rb_economic', 'rb_own']

    run_id = str(uuid.uuid4())
    experiments = []

    for fold, algorithm in itertools.product(baseline_folds, baseline_algorithms):
        experiments.append(
            Experiment(run_id=run_id, fold=fold[0], model_id=algorithm, train_data=fold[1], test_data=fold[2]))

    for fold, algorithm in itertools.product(forecasted_folds, other_algorithms):
        experiments.append(
            Experiment(run_id=run_id, fold=fold[0], model_id=algorithm, train_data=fold[1], test_data=fold[2]))
    run_experiments(experiments=experiments)
