from astar_ged.src.utils import get_root_path 
import nx_to_gxl
#import distance as dt
import networkx as nx



def create_graph(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    return g

g0 = create_graph([(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (5, 6), \
                   (6, 7), (7, 5)])
g1 = create_graph([(0, 1), (2, 3)])

g0.graph['gid']=0
g1.graph['gid']=1

nx.set_node_attributes(g0, 'type', {0: 'C', 1: 'O', 2:'N', 3:'N', 4:'N', 5:'N', 6:'N', 7:'N'})
#nx.set_node_attributes(g0, 'label',{0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7})
nx.set_node_attributes(g1, 'type', {0: 'O', 1:'N', 2:'C', 3: 'N'})
#nx.set_node_attributes(g1, 'label',{0:0, 1:1, 2:2, 3:3})

#nx.set_node_attributes(g0, 'strawberry', {0: -0.01, 1: 3.61423, 2: 1423})

'''
nx.set_edge_attributes(g0, 'some_label_1', {(0, 1): -1, (0, 2): -5, (7, \
                                                                     5): 2.0})

nx.set_edge_attributes(g0, 'some_label_2', {(0, 1): 1, (0, 2): 5, (7, 5): 2})

nx.set_edge_attributes(g0, 'some_label_3', {(0, 1): True, (0, 2): False})

nx.set_edge_attributes(g0, 'some_label_4', {(0, 1): True, (0, 2): -0.9, \
                                            (4, 5): 'xxx', (7, 5): '', \
                                            (5, 6): None})

nx.set_edge_attributes(g0, 'some_label_5', {(0, 1): True, (0, 2): None})

nx.set_edge_attributes(g0, 'some_label_6', {(0, 1): True, (0, 2): [False]})
'''
'''
print(g0.nodes(data=True))
print(g1.nodes(data=True))
print("="*50)
print(g0.edges(data=True))
print(g1.edges(data=True))
'''

nx_to_gxl(g0, 'g0', get_root_path() + '/files/g0.gxl')
#nx_to_gxl(g1, 'g1', get_root_path() + '/files/g1.gxl')

#d=dt.ged(g0, g1, 'astar', debug=False, timeit=False)
#print(d)
