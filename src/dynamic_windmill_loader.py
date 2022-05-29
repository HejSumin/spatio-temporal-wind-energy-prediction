import json
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from sklearn.preprocessing import minmax_scale

class DynamicWindmillLoader(object):
    """
    Loads the energy output of a cluster of wind turbines.
    This loader is designed to load a dataset with alternating graph topology (i.e., edges and edge weights) across all time periods.
    It transforms a json format graph file in to  DynamicGraphTemporalSignal type data iterator.

    This class is created based on "torch_geometric_temporal.dataset.encovid"
    (Reference: https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/dataset/encovid.html#EnglandCovidDatasetLoader)
    """
    def __init__(self, path):
        self._read_data(path)

    def _read_data(self, path):
        with open(path) as f:
            self._dataset = json.loads(f.read())

    def _get_edges(self):
        self._edges = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edges.append(
                np.array(self._dataset["edge_mapping"]["edge_index"][str(time)]).T
            )

    def _get_edge_weights(self):
        self._edge_weights = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edge_weights.append(
                np.array(self._dataset["edge_mapping"]["edge_weights"][str(time)])
            )

    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["y"])
        standardized_target = minmax_scale(stacked_target)
        # Energy output label "y" is normalized using min-max normalization when loaded.
        self.features = [
            standardized_target[i: i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]

    def get_dataset(self, lags: int = 4) -> DynamicGraphTemporalSignal:
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = DynamicGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset