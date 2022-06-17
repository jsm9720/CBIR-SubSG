import sys
import networkx as nx
import matplotlib.pyplot as plt


def create_graph(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    return g


def imgShow(nexG):
    nx.draw(nexG, with_labels=True)
    plt.show()


G = create_graph([(0, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)])

limit = 4
subs = []
for i in G.nodes():
    sub = [i]
    cur = i
    while True:
        l = list(G.neighbors(cur))
        if len(l) < limit-len(sub):
            sub.extend(l)
            cur = sub[-1]
        else:
            print(sub)
            print(list(set(l)-set(sub)))
            sys.exit()

imgShow(G)
