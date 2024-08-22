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

import copy
import csv
import datetime
import json
import numpy as np
import os
import pandas as pd
import requests
from tqdm import tqdm

# Remote source for downloading the load data
URL_LOAD_DATA = 'https://zenodo.org/records/3855575/files/Gesamtverbrauchsdaten.csv?download=1'

# Remote source for downloading the energy price data
URL_PRICE_DATA = 'https://www.smard.de/nip-download-manager/nip/download/market-data'

# Remote source and API token for downloading the PV generation data
URL_API_GENERATION_DATA = 'https://www.renewables.ninja/api/'
TOKEN_API_GENERATION_DATA = '<YOUR API TOKEN>'

# Time horizon
HORIZON = 168
# Number of households
N_HOUSEHOLDS = 200

PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}% {bar} | {elapsed}<{remaining}"


def _fetch_load_data() -> pd.DataFrame:
    """
    Fetch load data of 200 German households in 2019 from Zenodo.

    The data originates from
        Adrian Beyertt, Paul Verwiebe, Stephan Seim, Filip Milojkovic, and Joachim
        Müller-Kirchenbauer. 2020. Felduntersuchung zu Behavioral Energy Efficiency
        Potentialen von privaten Haushalten.

    :rtype: pd.DataFrame
    :return: Load data in hourly resolution.
    """
    print("Starting download...")
    df = pd.read_csv(URL_LOAD_DATA, sep=';', index_col=None)
    df = df.reset_index()

    # Aggregating with hourly resolution
    data = pd.DataFrame(columns=df.columns)
    for i in tqdm(range(0, len(df), 4), desc="Fetching load data", bar_format=PROGRESS_BAR_FORMAT):
        tmp = pd.DataFrame(columns=df.columns)
        tmp.loc[0, 'rec_time'] = df.loc[i, 'rec_time']
        tmp.loc[0, 'Einheit'] = 'kWh'
        tmp.iloc[0, 2:] = df.iloc[i:i + 4, 2:].sum().transpose()
        data = pd.concat([data, tmp])

    # Aligning timestamps (due to daylight saving time)
    data = data.reset_index()
    data = data.drop(['index'], axis=1)
    data = data[:365 * 24]
    data['timestamp'] = np.zeros(len(data))
    for i in range(len(data)):
        if '+01:00' in data.loc[i, 'rec_time']:
            data.loc[i, 'timestamp'] = datetime.datetime.strptime(data.loc[i, 'rec_time'],
                                                                  '%Y-%m-%d %H:%M:%S+01:00')
        else:
            data.loc[i, 'timestamp'] = datetime.datetime.strptime(data.loc[i, 'rec_time'],
                                                                  '%Y-%m-%d %H:%M:%S+02:00') - datetime.timedelta(
                hours=1, minutes=0)

    # Dropping irrelevant features
    data = data.drop(['rec_time', 'Einheit'], axis=1)
    return data


def _fetch_generation_data() -> pd.DataFrame:
    """
    Fetch generation data of a PV system located in German in 2019 from Renewables.ninja.

    The data is based on NASA's MERRA-2 dataset and was simulated in
        Stefan Pfenninger and Iain Staffell. 2016. Long-term patterns of European PV
        output using 30 years of validated hourly reanalysis and satellite data.
        Energy 114 (2016), 1251–1265.

    :rtype: pd.DataFrame
    :return: PV generation data in hourly resolution.
    """
    request = requests.session()
    request.headers = {'Authorization': 'Token ' + TOKEN_API_GENERATION_DATA}

    url = URL_API_GENERATION_DATA + 'data/pv'

    args = {
        'lat': 51.1638,
        'lon': 10.4478,
        'date_from': '2019-01-01',
        'date_to': '2019-12-31',
        'dataset': 'merra2',
        'capacity': 1.0,
        'system_loss': 0.1,
        'tracking': 0,
        'tilt': 35,
        'azim': 180,
        'format': 'json'
    }

    response = request.get(url, params=args)
    parsed_response = json.loads(response.text)

    data = pd.read_json(json.dumps(parsed_response['data']), orient='index')
    data.columns = ['capacity_factor']

    for i in tqdm(range(len(data)), desc="Fetching PV gen. data", bar_format=PROGRESS_BAR_FORMAT):
        tmp = ""

    data = data.reset_index()
    return data


def _fetch_price_data() -> pd.DataFrame:
    """
    Fetch German day-ahead wholesale electricity market prices in 2019 from

        Bundesnetzagentur für Elektrizität, Gas, Telekommunikation, Post und Eisenbahnen (www.smard.de).

    We apply an affine linear transformation to obtain consumer energy price data.

    :rtype: pd.DataFrame
    :return: Consumer energy price data in hourly resolution.
    """
    request = requests.session()
    request.headers = {'Content-Type': 'application/json'}

    content = {
        "request_form": [
            {
                "format": "CSV",
                "moduleIds": [
                    8004169,
                    8004170,
                    8000251,
                    8005078,
                    8000252,
                    8000253,
                    8000254,
                    8000255,
                    8000256,
                    8000257,
                    8000258,
                    8000259,
                    8000260,
                    8000261,
                    8000262,
                    8004996,
                    8004997
                ],
                "region": "DE",
                "timestamp_from": 1546297200000,
                "timestamp_to": 1577833199999,
                "type": "discrete",
                "language": "de",
                "resolution": "hour"
            }
        ]
    }

    response = request.post(URL_PRICE_DATA, json=content)
    reader = csv.reader(response.text.splitlines(), skipinitialspace=True)

    # Caching in a temporary file
    with open('data/tmp.csv', 'w', encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerows(reader)

    df = pd.read_csv('data/tmp.csv', sep=';')
    prices = []

    # Applying the affine linear transformation to obtain consumer energy prices
    for price in tqdm(df['Deutschland/Luxemburg [€/MWh] Originalauflösungen'], desc="Fetching price data",
                      bar_format=PROGRESS_BAR_FORMAT):
        price = float(price.replace(',', '.')) / 1000 + 0.25
        prices.append(price)

    # Remove temporary file
    os.remove('data/tmp.csv')

    data = pd.DataFrame()
    data['price'] = prices
    return data


def download_data() -> pd.DataFrame:
    """
    Download and aggregate load, PV generation, and energy price data from various remote sources.

    :rtype: pd.DataFrame
    :return: Aggregated dataset in hourly resolution (containing load, PV generation,and energy price data)
    """
    try:
        if not os.path.exists("data"):
            os.makedirs("data")

        load_data = _fetch_load_data()
        generation_data = _fetch_generation_data()
        price_data = _fetch_price_data()

        column_names = list(load_data.columns.values)[-1:] + list(load_data.columns.values)[:-1]

        data = pd.DataFrame(load_data[column_names])
        data['pv_capacity_factor'] = generation_data['capacity_factor']
        data['energy_price'] = price_data['price']
        return data
    except Exception as e:
        print("Something went wrong.\n", e)


def save_data(data: pd.DataFrame, is_forecast: bool = False) -> None:
    """
    Save the aggregated dataset to disk.

    :param pd.DataFrame data: Aggregated dataset (containing load, PV generation,and energy price data).
    :param bool is_forecast: Whether the dataset should be saved to disk with forecasts (feature values shifted by 1) or not. Default is False.
    """
    data = copy.deepcopy(data)
    suffix = ""

    # Shifting the feature values by 1 if the forecasts are to be saved
    if is_forecast:
        suffix += "-fc"
        data[data.columns[1:]] = data[data.columns[1:]].shift(periods=1)

    try:
        data = data[24:]
        data = data.reset_index()
        for n in tqdm(range(1, N_HOUSEHOLDS + 1), desc="Saving data to disk", bar_format=PROGRESS_BAR_FORMAT):
            id_household = "%03d" % (n,)
            output_dir = 'data/2019-' + str(HORIZON) + suffix + '/' + id_household + '/'
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            annual_consumption = 0
            for i in range(0, len(data) - HORIZON + 1, HORIZON):
                df_tmp = data.loc[i:i + HORIZON - 1, ['timestamp', str(n), 'pv_capacity_factor', 'energy_price']]
                df_tmp.columns = ['timestamp', 'load', 'pv_capacity_factor', 'energy_price']
                annual_consumption += np.sum(df_tmp['load'])

                df_tmp.to_csv(path_or_buf=output_dir + 'DE-2019_loads_pv_prices_' + id_household + '_' + str(
                    int(i / HORIZON)) + '.csv', index=False)

            # Saving additional information (i.e., the annual consumption) to a JSON file.
            info = {'annual_consumption': annual_consumption}
            with open(output_dir + 'DE-2019_info_' + id_household + '.json', 'w') as file:
                json.dump(info, file)
    except Exception as e:
        print("Something went wrong.\n", e)


if __name__ == "__main__":
    print("+++ DATA DOWNLOAD +++")
    data = download_data()
    if data is not None:
        print("+++ DATA PROCESSING +++")
        save_data(data=data, is_forecast=False)
        save_data(data=data, is_forecast=True)
        print("Completing processing ...")
    else:
        print("Download was not successful.")
