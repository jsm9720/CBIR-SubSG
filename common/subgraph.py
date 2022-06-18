#%%
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

def create_graph(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    return g

def img_Show(nexG):
    nx.draw(nexG, with_labels=True)
    plt.show()

def make_subgraph(graph, max_node):
    subs = []
    for i in graph.nodes():
        sub = [i]
        cur = i
        while True:
            neig = list(graph.neighbors(cur))
            neig = list(set(neig)-set(sub))
            space = max_node-len(sub)
            if len(neig) == 0:
                subs.append(tuple(tmp))
                break
            elif len(neig) <= space:
                sub.extend(neig)
                cur = sub[-1]
                if len(neig) == space:
                    subs.append(tuple(tmp))
                    break
            else:
                for c in combinations(list(neig),space):
                    tmp=sub.copy()
                    tmp.extend(list(c))
                    tmp.sort()
                    subs.append(tuple(tmp))
                break
    subgraphs = [graph.subgraph(i) for i in set(subs)]
    return subgraphs

def main():
    G = create_graph([(0, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)])
    # G = create_graph([(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6)])

    subgraphs = make_subgraph(G, 6)

    for i in range(len(subgraphs)):
        print(subgraphs[i])
        # imgShow(subgraphs[i])
    print(len(subgraphs))
    imgShow(G)

if __name__ == "__main__":
    main()

# %%
