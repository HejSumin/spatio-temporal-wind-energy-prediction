import json
import numpy as np
from math import sqrt
from tqdm import tqdm


def calculate_node_distance(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return sqrt(dx * dx + dy * dy)

def calculate_distance_weights(distance, _USE_EXPONENTIAL_PRIOR, _LAMBDA):
    if distance == 0:
        weight = 0
    else:
        if _USE_EXPONENTIAL_PRIOR:
            weight = np.exp(-(distance / _LAMBDA))
        else:
            weight = 1 / distance
    return weight

def create_output_blocks(data):
    blocks = []
    timestamps = data['TIME_CET'].unique()

    print("creating blocks ...")
    for i in tqdm(range(len(timestamps))):
        df = data.loc[data['TIME_CET'] == timestamps[i]]
        inner_list = []
        for index, row in df.iterrows():
            inner_list.append(row['VAERDI'])

        blocks.append(inner_list)
    return blocks

def calculate_joined_weight(delta_hat, dist_weight, _ALPHA, _BETA):
    return ((_ALPHA * (1 / delta_hat)) + (_BETA * dist_weight)) / (_ALPHA + _BETA)

def normalize_list(list):
    max_value = max(list)
    min_value = min(list)
    for i in range(len(list)):
        list[i] = (list[i] - min_value) / (max_value - min_value)
    return list

def save_np(blocks, path):
    np.save(path, blocks)


class NpEncoder(json.JSONEncoder):
    # Reference: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)