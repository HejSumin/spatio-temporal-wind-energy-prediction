import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from src.utility import NpEncoder
import src.utility as util

"""
Parameters

_SELF_WEIGHT: Edge weights for self-loops can be adjusted between 0-1.
_THETA: The threshold value for the creation of edges. If the edge weight is higher than the threshold, an edge is generated. 
_USE_EXPONENTIAL_FUNCTION: If set to True, the exponential function is used to obtain the edge weights.
_LAMBDA: A value to avoid numpy overflow errors when using the exponential function.
_ALPHA: Ratio of delta_hat value when calculating joined edge weights.
_BETA: Ratio of dist_weight value when calculating joined edge weights.
"""
_SELF_WEIGHT = 1.0
_THETA = 0.01
_USE_EXPONENTIAL_FUNCTION = False
_LAMBDA = 10000
_ALPHA = 0
_BETA = 1


class WindmillDynamicGraph(object):
    """
    Generating a static graph representation of wind turbines in a json format.
    This class uses a dataset containing a time series of wind energy data together with turbine location information as an input.
    """
    def __init__(self, path):
        self.save_as_file(path)

    def save_as_file(self, filepath):
        data = pd.read_csv('../../data_v2/rand10_merged.csv')
        y = util.create_output_blocks(data)
        util.save_np(y, "../../data_v2/rand10")
        # y = np.load("../../data_v2/rand10.npy")
        time_periods = data['TIME_CET'].unique()
        edge_index, edge_weights = self.create_edges_all(data, y, time_periods)

        edges_set = {}
        edge_weights_set = {}
        for time in range(len(time_periods)-1):
            edges_set[str(time)] = edge_index[time]
            edge_weights_set[str(time)] = edge_weights[time]

        edge_mapping = {}
        edge_mapping["edge_index"] = edges_set
        edge_mapping["edge_weights"] = edge_weights_set

        formatted = {}
        formatted["edge_mapping"] = edge_mapping
        formatted["time_periods"] = len(time_periods)
        formatted["y"] = y

        with open(filepath, 'w') as f:
            json.dump(formatted, f, cls=NpEncoder)

    def create_edges_all(self, data, output, time_periods):
        # Creating edges for all time periods
        delta_arr = self.compare_output(output)
        all_edges = []
        all_joined_weights = []
        edge_mapping = {}
        for i in range(len(time_periods)-1):
            e = self.create_onetime_edges(data, delta_arr, i)
            all_edges.append(e[0])
            all_joined_weights.append(e[1])
            edge_mapping[str(i)] = []

        return all_edges, all_joined_weights

    def create_onetime_edges(self, data, delta_arr, T):
        # Creating edges for a single timestamp
        # This method is called by create_edges_all and iterated for time_periods amount of time
        joined_weights = []
        edges = []
        self_loops = []
        self_weights = []
        vertices = data['GSRN'].unique()
        for i in tqdm(range(len(vertices)), position=0, leave=True):
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
                    delta_hat = np.abs(delta_arr[T][i] - delta_arr[T][j])
                    distance = util.calculate_node_distance(x1, y1, x2, y2)

                    dist_weight = util.calculate_distance_weights(distance, _USE_EXPONENTIAL_FUNCTION, _LAMBDA)
                    joined_weight = util.calculate_joined_weight(delta_hat, dist_weight, _ALPHA, _BETA)

                    i_list.append(i)
                    i_list.append(j)
                    joined_weights.append(joined_weight)
                    edges.append(i_list)
        edges = [e for e in edges if e != []]

        # NOTE: self_loops and self_weights are added after the edge weights are normalized between 0-1 to not to be affected by the normalization
        norm_edge_weights = util.normalize_list(joined_weights)

        clean_edges = []
        clean_edge_weights = []

        # Edges that have edge weights bigger than the threshold are selected in this step.
        for i in range(len(joined_weights) - 1):
            if joined_weights[i] >= _THETA:
                clean_edges.append(edges[i])
                clean_edge_weights.append(norm_edge_weights[i])

        clean_edges = clean_edges + self_loops
        clean_edge_weights = clean_edge_weights + self_weights

        # NOTE: If an edge weight value is NaN, the value is set to 0.0 to avoid a type error when training the model.
        np.nan_to_num(clean_edge_weights, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        return clean_edges, clean_edge_weights


    def compare_output(self, output):
        # Using the output block as an input, and returning the output difference between timestamps as a delta array.
        delta_arr = []
        for i in range(len(output) - 1):
            delta = self.calc_delta(np.array(output[i]), np.array(output[i + 1]))
            delta[np.isnan(delta)] = 0
            delta_arr.append(delta)
        return delta_arr

    def calc_delta_hat(self, delta1, delta2):
        # Calculating the difference between the two delta values as an absolute value
        return np.abs(delta1 - delta2)

    def calc_delta(self, output_t1, output_t2):
        # Calculating the difference in output values between two timestamps as a percentage
        np.seterr(divide='ignore', invalid='ignore')
        return (output_t2-output_t1) / np.abs(output_t1) * 100