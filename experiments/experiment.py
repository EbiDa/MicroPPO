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
import os.path
import pandas as pd
from datetime import datetime
from typing import List

from experiments.config import set_seed, EXP_SEED
from experiments.model_factory import ModelFactory, DummyModel


class Experiment:
    def __init__(self, run_id: str, fold: int, model_id: str, train_data: List[str], test_data: List[str]):
        self.run_id = run_id
        self.fold = fold
        self.model_id = model_id
        self.train_data = train_data
        self.test_data = test_data

        self.model: DummyModel = None
        set_seed(EXP_SEED)

    def run(self) -> None:
        self.model = ModelFactory().create(model_id=self.model_id, data_sample=self.train_data[0])
        self.model.set_run_id(run_id=self.run_id)
        self.model.train(datasets=self.train_data, fold_id=self.fold)
        results = self.model.evaluate(datasets=self.test_data, fold_id=self.fold)
        self._record(results=results)

    def _record(self, results: pd.DataFrame) -> None:
        date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d")

        output_path = "output/experiments/" + date + "/" + self.run_id + "/" + self.model.id

        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        except FileExistsError:
            print("Output folder already exists.")
        results.to_csv(output_path + "/" + self.model.id + "_" + timestamp + "_" + str(self.fold) + ".csv", sep=',',
                       mode='a')
