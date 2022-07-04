from common.subgraph import make_subgraph
from astar_ged.src.distance import ged, normalized_ged

import multiprocessing as mp
import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt


def make_pkl(dataset, queue, idxDict, train_num_per_row, max_row_per_worker):   # 64
    g1_list = []
    g2_list = []
    geds = []
    cnt = 0
    length = len(dataset)
    while True:
        if queue.empty():
            break
        num = queue.get()
        if length-num > max_row_per_worker:
            s = num
            e = num + max_row_per_worker
        else:
            s = num
            e = len(dataset)
        for i in range(s, e):
            for _ in range(train_num_per_row):
                dataset[i].graph['gid'] = 0
                # print(i, dataset[i])
                if cnt > (train_num_per_row//2):
                    a, b = idxDict[i]
                    # print(a, b)
                    r = random.randrange(a, b)
                    # print(1, r)
                else:
                    r = random.randrange(length)
                    # print(2, r)
                dataset[r].graph['gid'] = 1
                d = ged(dataset[i], dataset[r], 'astar',
                        debug=False, timeit=False)
                d = normalized_ged(d, dataset[i], dataset[r])
                g1_list.append(dataset[i])
                g2_list.append(dataset[r])
                geds.append(d)
                cnt += 1
            cnt = 0
        with open("common/data/DB_dataset_ver3_10000/{}_{}.pickle".format(s, e), "wb") as fw:
            pickle.dump([g1_list, g2_list, geds], fw)
        g1_list = []
        g2_list = []
        geds = []


def main():
    mp.set_start_method('spawn')
    q = mp.Queue()
    train_num_per_row = 64  # 64, 한 image가 비교하는 개수
    max_row_per_worker = 50  # 50, image 개수
    number_of_worker = 80  # 80, 프로세서 개수
    with open("data/networkx_ver3_10000.pickle", "rb") as fr:
        dataset = pickle.load(fr)
    total = []
    idxDict = dict()
    idx = [0]
    idx2 = []
    for i in range(len(dataset)):
        subs = make_subgraph(dataset[i], 3, False, False)
        idx2.append(len(subs))
        idx.append(len(subs)+idx[i])
        idxDict.update({j: (idx[i], idx[i+1])
                        for j in range(idx[i], idx[i+1])})
        total.extend(subs)
    # print("각 이미지에 대한 subgraph 수 :", idx2)
    # print("max", max(idx2), "min", min(idx2))
    # print("10개 이상 :", len([i for i in idx2 if 10 < i]))
    # print("20개 이상 :", len([i for i in idx2 if 20 < i]))
    # print("30개 이상 :", len([i for i in idx2 if 30 < i]))
    # print("50개 이상 :", len([i for i in idx2 if 50 < i]))
    # print("100개 이상 :", len([i for i in idx2 if 100 < i]))
    # print("총 subgraph 수 :", len(total))
    # exit()
    for i in range(0, len(total), max_row_per_worker):
        # for i in range(0, 10, max_row_per_worker): # test
        q.put(i)

    # print(total[0].nodes.data())
    # print(type(list(total[0].nodes.data())[0][1]['f1']))

    workers = []
    for i in range(number_of_worker):
        worker = mp.Process(target=make_pkl, args=(
            total, q, idxDict, train_num_per_row, max_row_per_worker))
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()


if __name__ == "__main__":
    main()
