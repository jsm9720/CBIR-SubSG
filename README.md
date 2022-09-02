# CBIR-SubSG
Contents based Image Retrieval Method based on Scene Graph using Subgraph Learning.   
Main idea is Subgraph Learning and A* Graph Edit Distance.   
Inspired by [Neural subgraph matching](http://snap.stanford.edu/subgraph-matching/) and [simGNN](https://arxiv.org/abs/1808.05689).   
- Subgraph Learning
  - Through GNN Learning, a similar pair of subgraphs are embedded close to the same embedding space
- A* Graph Edit Distance
  - Graph Similarity method that appllied with A* Algorithm to GED

## Train GNN encoder
1. Train the encoder : `python3 -m subgraph_matching.train`.    
Note that a trained order embedding model checkpoint is provided in `ckpt/model.pt`.
2. Optionally, analyze the trained encoder via `python3 -m subgraph_matching.test`

## Usage
Scene Graph data (`common/data.py`) can be used to train the model.   
Transfer the learned model to make inference on scene graph data (see `subgraph_matching/test.py`).

Available configurations can be found in `subgraph_matching/config.py`.

## Dependencies
The library uses PyTorch and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) to implement message passing graph neural networks (GNN). 
It also uses [DeepSNAP](https://github.com/snap-stanford/deepsnap), which facilitates easy use
of graph algorithms (such as subgraph operation and matching operation) to be performed during training for every iteration, 
thanks to its synchronization between an internal graph object (such as a NetworkX object) and the Pytorch Geometric Data object.

Graph Edit Distance(GED) uses graph-matching-toolkit used SimGNN papers.  
To use the GED, follow these steps:  
- `cd aster_ged/src && git clone https://github.com/jsm9720/graph-matching-toolkit.git`  
- Follow the instructions on https://github.com/jsm9720/graph-matching-toolkit to compile  
- java

Detailed library requirements can be found in requirements.txt   
