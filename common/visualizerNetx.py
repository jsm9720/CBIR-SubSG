'''
만든이 : 이하은
개요 :
그래프를 visual genome에 있는 이미지와 매칭 시켜 plot 해주는 코드
코드를 실행하기 위해서는 visual_genome_python_driver가 필요
'''
from visual_genome_python_driver.visual_genome import api as vg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image
import networkx as nx
import requests
import json
import pickle
import sys
import time
import numpy

'''
    Subgraph의 node attribute 중 origin Id를 통해 
    Subgraph와 대상 그래프 간 Object Name값이 일치하는 Node를 
    Img(대상 그래프)에서 Bounding Box를 통해 확인
'''

'''
with open("data/networkx_ver2.pickle", "rb") as fr:
    graphs = pickle.load(fr)
'''

try:
    from StringIO import StringIO as ReadBytes
except ImportError:
    print("Using BytesIO, since we're in Python3")
    # from io import StringIO #..
    from io import BytesIO as ReadBytes


with open('common/data/scene_graphs.json') as file:  # open json file
    sceneJson = json.load(file)


def makeObjectsInSubG(gId, G):
    '''
    그래프 내 모든 노드의 이미지 내 위치값과 이름을 갖는 Dict List 반환
    '''
    # 0번 scenegraph의
    objects = sceneJson[gId]['objects']

    # ObjId : coordinate(x,y,w,h, name)
    IdCoNameDict = {}

    for i in range(len(objects)):
        object_id = objects[i]['object_id']
        obj = objects[i]
        if len(obj['synsets']) != 0:
            IdCoNameDict[object_id] = {
                'x': obj['x'], 'y': obj['y'], 'w': obj['w'], 'h': obj['h'], 'name': obj['names'][0], 'synsets': obj['synsets'][0].split('.')[0]}
        else:
            IdCoNameDict[object_id] = {
                'x': obj['x'], 'y': obj['y'], 'w': obj['w'], 'h': obj['h'], 'name': obj['names'][0], 'synsets': obj['names'][0]}

    nodes = G.nodes(data="originId")
    nodes = [n[1] for n in nodes]
    objectList = []
    for idx in nodes:
        objectList += [IdCoNameDict[idx]]

    return objectList


def mkDenseObjInSubG(gId, G):
    '''
    그래프 내 이웃 노드 수가 가장 많은 노드와 그 이웃 노드의 이미지 내 정보를 갖는 Dict List 반환
    '''

    # 0번 scenegraph의
    objects = sceneJson[gId]['objects']

    # ObjId : coordinate(x,y,w,h, name)
    IdCoNameDict = {}

    for i in range(len(objects)):
        object_id = objects[i]['object_id']
        obj = objects[i]
        if len(obj['synsets']) != 0:
            IdCoNameDict[object_id] = {
                'x': obj['x'], 'y': obj['y'], 'w': obj['w'], 'h': obj['h'], 'name': obj['names'], 'synsets': obj['synsets'].split('.')[0]}
        else:
            IdCoNameDict[object_id] = {
                'x': obj['x'], 'y': obj['y'], 'w': obj['w'], 'h': obj['h'], 'name': obj['names'], 'synsets': obj['name']}

    # nodes = G.nodes(data="originId")
    # nodes = [n[1] for n in nodes]
    denseNode = []
    dNode = 0

    for i in G.nodes:
        if (len(list(G.neighbors(i))) >= 5):
            denseNode.append(i)
        elif (len(list(G.neighbors(i))) >= dNode):
            dNode = i

    if (dNode in denseNode) != 0:
        denseNode.append(dNode)
    objectList = []
    [[objectList.append(IdCoNameDict[objects[idx]['object_id']]) for idx in G.neighbors(nodeIdx)] for nodeIdx in
     denseNode]

    denseNodeList = []
    [denseNodeList.append(IdCoNameDict[objects[idx]['object_id']])
     for idx in denseNode]

    return denseNodeList, objectList


# visualize with bounding box(use Visual Genome api)
def patchOnImgApi(image, objectsList, denseNode, filePath):
    response = requests.get(image.url)
    img = PIL_Image.open(ReadBytes(response.content))
    plt.imshow(img)
    ax = plt.gca()
    if len(denseNode) != 0:
        for object in denseNode:
            ax.add_patch(Rectangle((object['x'], object['y']),
                                   object['w'], object['h'],
                                   fill=False,
                                   edgecolor='green', linewidth=3))
            ax.text(object['x'], object['y'], object['synsets'], style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

    for object in objectsList:
        ax.add_patch(Rectangle((object['x'], object['y']),
                               object['w'], object['h'],
                               fill=False,
                               edgecolor='red', linewidth=3))
        ax.text(object['x'], object['y'], object['synsets'], style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    fig = plt.gcf()
    #plt.tick_params(labelbottom='off', labelleft='off')
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.savefig(filePath+".png")

# visualize with bounding box(Local Image File)


def patchOnImgLocal(imagepath, objectsList):

    img = PIL_Image.open(imagepath)
    plt.imshow(img)
    ax = plt.gca()
    for object in objectsList:
        ax.add_patch(Rectangle((object['x'], object['y']),
                               object['w'], object['h'],
                               fill=False,
                               edgecolor='red', linewidth=3))
        ax.text(object['x'], object['y'], object['synsets'], style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    fig = plt.gcf()
    #plt.tick_params(labelbottom='off', labelleft='off')
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.show()


def plotOnImg(ImgId, G, path):

    image = vg.get_image_data(ImgId)

    plt.cla()
    plt.clf()

    # visualize on plt
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    '''
    # 전체 이미지 plot
    plt.subplot(1, 3, 1)
    # oriNameDict = {id: name for id, name in graphs[ImgId-1].nodes(data='name')}
    # graphs[ImgId-1] = nx.relabel_nodes(graphs[ImgId-1], oriNameDict)
    # nx.draw(graphs[ImgId-1], with_labels=True)
    '''
    plt.subplot(1, 3, 2)
    nameDict = {id: name for id, name in G.nodes(data='name')}
    G = nx.relabel_nodes(G, nameDict)
    nx.draw(G, with_labels=True)
    denseNode = []
    plt.subplot(1, 3, 3)
    objectList = makeObjectsInSubG(ImgId-1, G)
    #denseNode, objectList = mkDenseObjInSubG(gId, G)

    patchOnImgApi(image, objectList, denseNode, path)


if __name__ == "__main__":
    with open("data/networkx_ver2.pickle", "rb") as fr:
        graphs = pickle.load(fr)
    gId = 999
    G = graphs[gId]

    plotOnImg(gId, G)
