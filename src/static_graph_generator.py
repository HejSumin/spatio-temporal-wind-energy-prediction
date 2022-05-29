import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.preprocessing import minmax_scale
from src.utility import NpEncoder
import src.utility as util
import numpy as np

"""
Parameters

_SELF_WEIGHT: Edge weights for self-loops can be adjusted between 0-1.
_THETA: The threshold value for the creation of edges. If the edge weight is higher than the threshold, an edge is generated. 
_USE_EXPONENTIAL_FUNCTION: If set to True, the exponential function is used to obtain the edge weights.
_LAMBDA: A value to avoid numpy overflow errors when using the exponential function.
"""

_SELF_WEIGHT = 1.0
_THETA = 0.1
_USE_EXPONENTIAL_FUNCTION = True
_LAMBDA = 10000


class WindmillStaticGraph(object):
    """
    Generating a static graph representation of wind turbines in a json format.
    This class use two datasets as input: one containing wind turbine information, and another containing a time series wind energy dataset.
    """

    def __init__(self):
        self.merge_dataset("data/")
        # self.save_as_file(path)

    def merge_dataset(self, path):
        data_path = "data/random_sample_2y_agg1h.csv"
        master_path = path + "master2018.csv"

        data = pd.read_csv(data_path)
        master = pd.read_csv(master_path)

        merged = pd.merge(data, master, on="GSRN", how='inner')
        filtered = merged[["GSRN", "TIME_CET", "UTM_x", "UTM_y", "VAERDI"]]

        result = filtered.drop_duplicates(subset=['GSRN', 'TIME_CET', 'VAERDI'], keep='first')
        df = result.dropna()

        filepath = Path('data/merged.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)


    def save_as_file(self, filepath):
        data = pd.read_csv('data/merged.csv')
        block = np.load("data/rand60.npy")
        # block = util.create_output_blocks(data)
        # util.save_np(block, "data/rand60.npy")

        edges, weights = self.create_edges(data)

        formatted = {}
        formatted["block"] = block
        formatted["edges"] = edges
        formatted["weights"] = weights
        with open(filepath, 'w') as f:
            json.dump(formatted, f, cls = NpEncoder)

    def create_edges(self, data):
        edges = []
        edge_weights = []
        self_loops = []
        self_weights = []
        vertices = data['GSRN'].unique()
        for i in tqdm(range(len(vertices))):
            x1 = data.loc[data['GSRN'] == vertices[i], 'UTM_x'].mean()
            y1 = data.loc[data['GSRN'] == vertices[i], 'UTM_y'].mean()
            for j in range(len(vertices)):
                if(i == j):
                    self_arr = []
                    self_arr.append(i)
                    self_arr.append(j)
                    self_weights.append(_SELF_WEIGHT)
                    self_loops.append(self_arr)
                else:
                    i_list = []
                    x2 = data.loc[data['GSRN'] == vertices[j], 'UTM_x'].mean()
                    y2 = data.loc[data['GSRN'] == vertices[j], 'UTM_y'].mean()

                    distance = util.calculate_node_distance(x1, y1, x2, y2)
                    weight = util.calculate_distance_weights(distance, _USE_EXPONENTIAL_FUNCTION, _LAMBDA)

                    i_list.append(i)
                    i_list.append(j)
                    edge_weights.append(weight)
                    edges.append(i_list)

        # NOTE: To avoid them being affected by the normalization self_loops and self_weights are added after the edge weights are normalized between 0-1.
        norm_edge_weights = minmax_scale(edge_weights).tolist()

        clean_edges = []
        clean_edge_weights = []

        # Edges with edge weights above the threshold are selected in this step.
        for i in range(len(norm_edge_weights)-1):
            if norm_edge_weights[i] >= _THETA:
                clean_edges.append(edges[i])
                clean_edge_weights.append(norm_edge_weights[i])

        clean_edges = clean_edges + self_loops
        clean_edge_weights = clean_edge_weights + self_weights
        clean_edges = [e for e in clean_edges if e != []]

        return clean_edges, clean_edge_weights
