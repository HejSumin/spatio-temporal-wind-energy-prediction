from datetime import timedelta
import random
import pandas as pd
from pyproj import Proj
from pathlib import Path
from tqdm import tqdm

"""
Methods applied when the raw dataset needs some preparation prior to creating a graph structure.
"""

def turbines_in_bb(data, min_UTM_x, min_UTM_y, max_UTM_x, max_UTM_y):
    data = data[(data['UTM_x'] <= max_UTM_x) & (data['UTM_y'] <= max_UTM_y)]
    data = data[(min_UTM_x <= data['UTM_x']) & (min_UTM_y <= data['UTM_y'])].drop_duplicates(
        subset=['GSRN', 'UTM_x', 'UTM_y'], keep='first').dropna()
    return data['GSRN'].to_numpy()

def random_turbines(data, num_nodes):
    GSRN = data['GSRN'].unique()
    sample = random.sample(list(GSRN), num_nodes)
    return sample

def convert_to_utm(lon, lat):
    myProj = Proj("+proj=utm +zone=32")
    UTM_x, UTM_y = myProj(lon, lat)
    return UTM_x, UTM_y

def filter_with_bb(data, GSRN_arr):
    data = data.loc[data['GSRN'].isin(GSRN_arr)]
    filepath = Path('../../data/sjaelland/sjaelland_381.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath)
    return data

def filter_time(data, start_time, end_time):
    data['TIME_CET'] = pd.to_datetime(data['TIME_CET'], utc=True)
    data.sort_values(by='TIME_CET', inplace=True)
    mask = (data['TIME_CET'] >= start_time) & (data['TIME_CET'] <= end_time)
    filtered = data.loc[mask]
    timestamps = filtered['TIME_CET'].unique()
    gsrn_numbers = filtered['GSRN'].unique()

    print(len(gsrn_numbers))

    print("checking timestamps ...")
    # find GSRN instances that occur multiple times at the same timestamp and take mean value
    filtered = filtered.groupby(['GSRN', 'TIME_CET'], as_index=False).agg({'VAERDI': 'mean'})

    # check if GSRN instances are missing and take mean value of gsrn occurrence at previous and succeeding timestamp
    for i in tqdm(timestamps):
        df = filtered.loc[filtered['TIME_CET'] == i]
        previous_time = i - timedelta(hours=1)

        if len(df.index) < len(gsrn_numbers):
            print("if-statement ...")
            print(len(df.index), len(gsrn_numbers))
            # take value from previous time
            for gsrn in gsrn_numbers:
                if not df.isin([gsrn]).any().any():
                    value = filtered.loc[(filtered['TIME_CET'] == str(previous_time)) & (filtered['GSRN'] == gsrn), ['VAERDI']].iloc[0].iloc[0]
                    row = {'TIME_CET': i, 'GSRN': gsrn, 'VAERDI': value}
                    filtered = filtered.append(row, ignore_index=True)

                    check = filtered.loc[filtered['TIME_CET'] == i]
                    print(len(check.index), len(gsrn_numbers))
        else:
            continue

    filepath = Path('../../data/skagen/skagen_297_1year.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(filepath)
    return filtered


def aggregate_data(data, aggregation_interval):
    data = data[['TIME_CET', 'VAERDI', 'GSRN']]
    data['TIME_CET'] = pd.to_datetime(data['TIME_CET'], utc=True)
    aggregated_data = data.groupby('GSRN').resample(aggregation_interval, on='TIME_CET')['VAERDI'].sum()
    # the inverse of groupby, reset_index
    aggregated_data = aggregated_data.reset_index()
    # set again the timestamp as index
    aggregated_data = aggregated_data.set_index("TIME_CET")
    # default origin = 'start of day' which is the first day at midnight of the timeseries, perhaps we rather want the first value of the timeseries
    filepath = Path('../../data/skagen/skagen_297_1year_agg.csv')
    aggregated_data.to_csv(filepath)
    return aggregated_data
