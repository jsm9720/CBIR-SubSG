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

    def split(node, subs, max, sub=None):
        if sub==None:
            sub = [node]

        cur = node
        while True:
            neig = list(graph.neighbors(cur))
            neig = list(set(neig)-set(sub))
            space = max-len(sub)
            if len(neig) == 0:
                # 더이상 갈 곳이 없는 경우
                sub.sort()
                subs.add(tuple(sub))
                break
            elif len(neig) <= space:
                # 여러 곳으로 갈 수 있을 경우
                if len(neig) == 1:
                    sub.extend(neig)
                    cur = neig[0]
                else:
                    sub.extend(neig)
                    if len(neig) == space:
                        sub.sort()
                        subs.add(tuple(sub))
                        break
                    for i in neig:
                        cur = i
                        tmp = sub.copy()
                        split(cur, subs, max, tmp)
                    break

            else:
                # 갈 곳이 많지만 subgraph 노드 개수를 넘을 경우
                for c in combinations(list(neig),space):
                    tmp=sub.copy()
                    tmp.extend(list(c))
                    tmp.sort()
                    subs.add(tuple(tmp))
                break
    
    subgraphs = set()
    for i in graph.nodes():
        split(i, subgraphs, max_node)
    return [graph.subgraph(i) for i in set(subgraphs)]

def main():
    G = create_graph([(0, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)])
    # G = create_graph([(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6)])

    graphs = make_subgraph(G, 3)

    for i in range(len(graphs)):
        print(graphs[i])
        img_Show(graphs[i])
    print(len(graphs))
    img_Show(G)

if __name__ == "__main__":
    main()

# %%
