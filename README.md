# Spatio-temporal wind energy prediction using GNNs


## Tools


### Sample Dataset
A preprocessed 2-year wind power record from a European country. It consists of 1-hour intervals time series data. Used wind turbines are randomly selected to preserve the anonimity of the data. Due to the confidentiality, raw data is not included in this repository. 
The sample dataset was generated with a value `0.001` as threshold using `static_graph_generator.py`.


### Pytorch Geometric Temporal
The prediction models are implemented by using [Pytorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/index.html).


### Usage

- Library installation
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip install torch-geometric
pip install torch-geometric-temporal
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu113` depending on your PyTorch installation.

For more information on the library installation, please visit the documentation of [Pytorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/index.html). 

- To run the prediction test with the sample dataset, simply run the `DCRNN.ipynb` or `LRGCN.ipynb` notebooks. 
- To create a new graph format dataset, preprocess the raw dataset by using `dataset_preparation`, then use `static_graph_generator.py` or `dynamic_graph_generator.py`. 



## Authors
Anne Havmøller Fellows-Jensen (afel@itu.dk) & Sumin Lee (sule@itu.dk)


### Supervisor
Maria Sinziiana Astefanoaei
