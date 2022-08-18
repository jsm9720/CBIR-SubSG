import torch
import networkx as nx
import sys
import time
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils
from tqdm import tqdm

# a = -3
# b = -4
# dataset = TUDataset(root="/tmp/PROTEINS", name="PROTEINS")
# dataset = list(dataset)
# print(dataset[a])
# print(dataset[b])
# print(dataset[a].edge_index)
# data = []
# for i, graph in tqdm(enumerate(dataset)):
#     if not type(graph) == nx.Graph:
#         graph = pyg_utils.to_networkx(graph).to_undirected()
#         data.append(graph)
# print(data[a])
# print(data[b])
# print("-----------------------------")
# start = time.time()
# G = nx.optimize_graph_edit_distance(data[a], data[b])
# for i in G:
#     print(i)
# print("time : ", time.time()-start)

'''
# 첫번째 실험 [0개, 0.069초]
Data(edge_index=[2, 18], x=[5, 3], y=[1])
Data(edge_index=[2, 18], x=[5, 3], y=[1])

# 두번째 실험 [7개, 71140.000초//약 49시간]
Data(edge_index=[2, 58], x=[16, 3], y=[1])
Data(edge_index=[2, 54], x=[15, 3], y=[1])
'''

G1 = nx.cycle_graph(4)
G2 = nx.cycle_graph(4)
G3 = nx.Graph()

# G1.add_nodes_from([(0, {"f1": 2, "f2": 2})])
# G1.add_nodes_from([(1, {"f1": 2})])
G1.add_edges_from([(0, 1, {"f3": 2})])
G1.add_edges_from([(1, 2, {"f3": 2})])
G1.add_edges_from([(2, 3, {"f3": 2})])
G1.add_edges_from([(3, 0, {"f3": 2})])
# G1.relabel({1: 5, 2: 6})
# G1.add_nodes_from([(2, {"color": "white"}), (3, {"color": "green"})])

G3.add_nodes_from([5, 6, 7, 8])
# G3.add_nodes_from([(5, {"f1": 1}), 6, 7, 8, 9])
G3.add_edges_from([(5, 6), (6, 7), (7, 8)])
G3.add_edges_from([(5, 6, {"f1": 1}), (6, 7, {"f2": 1}),
                  (7, 8, {"f1": 1}), (8, 5, {"f1": 1})])
# G2.add_nodes_from([(0, {"color": "blue"}), (3, {"color": "green"})])
# G1.add_edges_from([(2, 3, {"color": "white"}), (1, 3, {"color": "green"})])
# G2.add_edges_from([(0, 3, {"color": "white"}), (1, 3, {"color": "green"})])

print(G1.nodes.data())
print(G2.nodes.data())
print(G3.nodes.data())
print("="*40)
print(G1.edges.data())
print(G2.edges.data())
print(G3.edges.data())
print("="*40)
# G2.add_node(10)
# G2.add_edge(1, 10)
print(G1)
print(G2)
print(G3)

print("G1,G2", nx.graph_edit_distance(G1, G2))
print("G1,G3", nx.graph_edit_distance(G1, G3))
