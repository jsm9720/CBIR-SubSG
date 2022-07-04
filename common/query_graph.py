'''
만든이:이하은
코드 개요: Query를 위한 Query graph 생성
'''
import sys
import numpy as np
import pandas as pd
import torch
import csv

import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import FastText
from tqdm import tqdm
import time
import json
from collections import Counter
import pickle
import nltk
from nltk.corpus import conll2000


def vGphShow(nexG):
    plt.figure(figsize=[15, 7])
    nx.draw(nexG, with_labels=True)
    plt.show()

# with open("data/networkx_ver2.pickle", "rb") as fr:
#     ver2G = pickle.load(fr)


with open("data/networkx_ver1.pickle", "rb") as fr:
    net200 = pickle.load(fr)


nodeNames = ['truck', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'road', 'bicycle', 'road',
             'window', 'building', 'car', 'tree', 'man', 'sign', 'sidewalk', 'street', 'truck', 'leaf', 'tree',
             'chair', 'rock', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'person', 'tree', 'chair',
             'person', 'flower', 'cloud', 'grass', 'land', 'bench', 'sky', 'rock', 'leaf', 'lamp', 'keyboard',
             'monitor', 'plate', 'car', 'car', 'road', 'light',
             'cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair',
             'keyboard', 'monitor', 'cabinet', 'cup', 'rug', 'curtain', 'desk', 'window', 'book', 'chair', 'wall',
             'window', 'table', 'light', 'book', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling',
             'table', 'light', 'book', 'desk', 'pillow', 'letter', 'book', 'cup', 'bottle', 'ceiling', 'paper',
             'man', 'shirt', 'bag', 'water', 'hat', 'mirror', 'seat', 'ceiling', 'leg', 'wall', 'person',
             'shirt', 'bag', 'water', 'bag', 'water', 'hat', 'mirror', 'seat', 'ceiling', 'leg', 'jacket']

nodeNames = list(set(nodeNames))


embDict = {}
for i in range(len(net200)):
    names = [row[1] for row in net200[i].nodes(data='name')]
    f0 = [row[1] for row in net200[i].nodes(data='f0')]
    f1 = [row[1] for row in net200[i].nodes(data='f1')]
    f2 = [row[1] for row in net200[i].nodes(data='f2')]

    for i in range(len(names)):
        features = [f0[i], f1[i], f2[i]]
        embDict[names[i]] = features


nodeNameList = [[['light', 'car', 'car'],
                ['car', 'road', 'tire']],
                [['tag', 'car', 'car'],
                 ['car', 'plate', 'road']]]


gList = []
for i in range(len(nodeNameList)):
    objNodeName = nodeNameList[i][0]
    subNodeName = nodeNameList[i][1]

    df = pd.DataFrame({"objNodeName": objNodeName,
                      "subNodeName": subNodeName, })
    gI = nx.from_pandas_edgelist(
        df, source='objNodeName', target='subNodeName')

    nodesList = objNodeName + subNodeName

    for index, row in df.iterrows():
        gI.nodes[row['objNodeName']]['name'] = row["objNodeName"]  # name attr
        gI.nodes[row['subNodeName']]['name'] = row['subNodeName']  # name attr

    for i in range(len(nodesList)):  # nodeId
        name = nodesList[i]
        emb = embDict[name]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
        for j in range(3):  # Embedding 값은 [3,]인데, 원소 각각을 특징으로 node에 할당
            nx.set_node_attributes(gI, {name: float(emb[j])}, "f" + str(j))

    dictIdx = {nodeId: idx for idx, nodeId in enumerate(nodesList)}
    gI = nx.relabel_nodes(gI, dictIdx)
    gList.append(gI)


with open("common/data/query01.pickle", "wb") as fw:
    pickle.dump(gList, fw)

# with open("./data/query01.pickle", "rb") as fr:
#     gList = pickle.load(fr)

# vGphShow(gList[0])
