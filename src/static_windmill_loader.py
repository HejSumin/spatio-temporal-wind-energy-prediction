import json
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from sklearn.preprocessing import minmax_scale

class StaticWindmillLoader(object):
    """
    Loads the energy output of a cluster of wind turbines.
    This loader is designed to load a dataset with static graph topology across all time periods.
    It transforms a json format graph file in to a StaticGraphTemporalSignal type data iterator.

    This class is created based on "torch_geometric_temporal.dataset.windmillsmall"
    (Reference: https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/dataset/windmillsmall.html)
    """

    def __init__(self, path):
        self._read_data(path)

    def _read_data(self, path):
        with open(path) as f:
            self._dataset = json.load(f)

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        stacked_target = np.stack(self._dataset["block"])
        # Energy output label "block" is normalized using min-max normalization when loaded.
        standardized_target = minmax_scale(stacked_target)
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
        # lags: the number of time lags
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
