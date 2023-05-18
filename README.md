
# GNNInterpreter Experiments

This code is based on [GNNInterpreter](https://arxiv.org/pdf/2209.07924.pdf)

Create a conda environment with necessary requirements using the following command:

```conda env create -f environment.yml```

- *explain.py* contains the GNNInterpreter code
- *probgraph.py* defines the graph that is optimized to explain a GNN model
- *gnn.py* contains the code for the GNN model described by the GNNInterpreter paper for use with the MUTAG dataset
- *main.py* trains a GNN model on the MUTAG dataset and generates explanations for both classes
- *unresolved.txt* contains a list of ambiguities in the original GNNInterpreter paper, as well as new directions for future work