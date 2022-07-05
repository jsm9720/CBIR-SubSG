# %%
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import sys
import pickle
import random


def create_graph(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    return g


def img_Show(nexG):
    nx.draw(nexG, with_labels=True)
    plt.show()


def make_subgraph(graph, max_node, train, R_BFS):

    def split(node, subs, max, train, R_BFS, sub=None):
        if train:
            s = max//2
            max = random.randrange(s, max)

        if sub == None:
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
                    if not R_BFS:
                        # 모든 상황 고려
                        for i in neig:
                            cur = i
                            tmp = sub.copy()
                            split(cur, subs, max, False, R_BFS, tmp)
                        break
                    else:
                        cur = random.choice(neig)
            else:
                # 갈 곳이 많지만 subgraph 노드 개수를 넘을 경우
                if not R_BFS:
                    for c in combinations(list(neig), space):
                        tmp = sub.copy()
                        tmp.extend(list(c))
                        tmp.sort()
                        subs.add(tuple(tmp))
                    break
                else:
                    # 교집합 부분으로 수정해야함
                    sub.extend(
                        list(random.choice(list(combinations(neig, space)))))
                    sub.sort()
                    subs.add(tuple(sub))
                    break

    subgraphs = []
    class_set = set()
    total_subs = set()
    for i in graph.nodes():
        split(i, total_subs, max_node, train, R_BFS)
    pre = [graph.subgraph(i) for i in total_subs]
    # 노드 클래스가 중복으로 가지는 subgraph filtering
    for j in pre:
        class_sub = tuple([f['name'] for _, f in list(j.nodes.data())])
        if len(set(class_sub)) == 1:
            continue
        elif class_sub not in class_set:
            subgraphs.append(j)
            class_set.add(class_sub)
            class_set.add(tuple(reversed(class_sub)))

    return subgraphs


def main():
    # G = create_graph([(0, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)])
    G = create_graph([(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6)])
    nx.set_node_attributes(
        G, {0: 'C', 1: 'O', 2: 'N', 3: 'N', 4: 'B', 5: 'N', 6: 'A'}, 'name')

    graphs = make_subgraph(G, 4, True)

    for i in range(len(graphs)):
        print(graphs[i])
        img_Show(graphs[i])
    print(len(graphs))


if __name__ == "__main__":
    main()

# %%
