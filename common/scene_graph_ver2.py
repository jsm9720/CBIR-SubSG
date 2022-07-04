'''
만든이 : 이하은
'''
import sys
import numpy as np
import pandas as pd
import torch
import csv

import torch_geometric.utils

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


def make_graph(train):

    # 먼저 변경하는 바람에 기존 originId를 이용해 Object의 위치를 알 수 없는 문제 발생 -> 해당 오류 정정 필요

    #
    # conllDict = {word: tag for word, tag in conll2000.tagged_words(tagset='universal')}
    # print(conllDict['ostirch'])
    #
    # sys.exit()
    '''
        objId를 기준으로 그래프를 생성하되, 
        1. 단일 node에 이웃 노드 이름이 5개 이상 겹치는 경우, 이름이 같은 오브젝트들을 모두 하나의 node로 묶고 오브젝트를 삭제함
        2. neighbor node의 범위가 중복되는 경우 id를 name 기준으로 합친 후 묶인 id 중에서 제일 첫번째의 id부여하고 (근처에 위치하는) 동일 이름을 가진 obj의 id값을 새로 부여한 id로 치환
        -> 첫번째의 id 값으로 node id를 모두 변경하는 List를 새로 만들어서, 
        3. 중요 object Name은 합치지 않도록 분기처리 할 것
    '''

    def get_key(dict, val):
        for key, value in dict.items():
            if val == value:
                return key
        return "key doesn't exist"

    '''
        Synset Naming
        2. noun이 아닐 때 NLTK를 통해서 명사를 찾아 name으로 사용 가능한(synset으로 대체 가능한) noun 추출
            -> noun이 두 개 일 경우 기존 synset List에서 가장 많이 사용되는 단어를 사용하고, 개수가 일치 할 경우 [0]의 단어를 name으로 함
            -> NLTK로 분할한 noun들이 기존 synset이 아닌 경우 전체를 통째로 synset으로 만들기
    '''

    def extractNoun(noun, synsDict, synNameCnter):
        conllDict = {word: tag for word,
                     tag in conll2000.tagged_words(tagset='universal')}
        words = noun.split(' ')

        # synset에 해당되는 noun 중 언급이 많은 단어 선택
        name = ''
        cnt = 0
        # noun 판별
        nouns = []
        for i in words:
            try:
                if conllDict[i] == 'NOUN':
                    nouns.append(i)
            except:
                continue

        # synset에 해당되는 noun이 있는지 판별
        nInSynsDictList = []

        if len(nouns) != 0:  # nouns이 있음
            for i in nouns:
                try:
                    nInSynsDictList.append(synsDict[i])
                except:
                    continue
                    # noun이 아예 없는 경우?

            if len(nInSynsDictList) != 0:
                for i in nInSynsDictList:
                    if cnt < synNameCnter[i]:
                        name = i
                        cnt = synNameCnter[i]
            else:
                if len(nouns) != 0:
                    name = "_".join(sorted(nouns))

        else:
            name = "_".join(sorted(words))

        return name

    gList = []
    imgCnt = 1000
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)

    # --------------------------- vvv synset Dict, Total Embedding(fasttext 값) vvv ---------------------------
    '''synset Name Dict'''

    synsetList = []
    originIdList = []
    nonSysnNameList = []
    nonSysnIdList = []
    originDict = {}

    for ImgId in range(imgCnt):
        imageDescriptions = data[ImgId]["objects"]
        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            oId = imageDescriptions[j]['object_id']
            try:
                synsetName = imageDescriptions[j]['synsets'][0].split(".")
                synsetList.append(synsetName[0])
                originIdList.append(str(oId))
                originDict[str(oId)] = imageDescriptions[j]['names'][0]

            except Exception:
                nonSysnNameList.append(imageDescriptions[j]['names'][0])
                nonSysnIdList.append(str(oId))

    synNameCnter = Counter(synsetList)

    '''   
        Synset Naming
        1. originDict{synset을 갖는 objId : objName}에서 nonSynsetName의 원소가 있는 경우, objId로 synsDict에서 synsetName을 찾음
        2. nonSynsetName/Id에서 해당 원소를 제외하고, synsDict에 추가함(objId : objName(synset))
        없는 경우, 2로 넘어감
    '''

    synsDict = {idx: name for idx, name in zip(originIdList, synsetList)}
    nonSynsDict = {name: value for name,
                   value in zip(nonSysnIdList, nonSysnNameList)}

    for i in range(len(nonSysnIdList)):
        try:
            # 1. originDict{synset을 갖는 objId : objName}에서 nonSynsetName의 원소가 있는 경우,
            sameNameId = get_key(originDict, nonSysnNameList[i])
            synsDict[str(nonSysnIdList[i])] = synsDict[str(sameNameId)]

        except:
            # 2. noun이 아닐 때 NLTK를 통해서 명사를 찾아 name으로 사용 가능한(synset으로 대체 가능한) noun 추출
            name = extractNoun(nonSysnNameList[i], synsDict, synNameCnter)
            synsDict[str(nonSysnIdList[i])] = name

    # #위에서 만든 synset Dict를 이용해 totalEmbedding 값을 만듦(fasttext)
    objectNameList = list(set(list(synsDict.values())))
    model, totalEmbDict = FeatEmbeddPerTotal_model(objectNameList)

    # print(totalEmbDict[synsDict['1058559']])

    with open("./data/totalEmbDict.pickle", "wb") as fw:
        pickle.dump(model, fw)

    # --------------------------- ^^^ synset Dict, Total Embedding(fasttext 값)^^^ ---------------------------

    for i in tqdm(range(imgCnt)):
        objId, subjId, relatiohship, edgeId, weight = AllEdges(data, i)
        objIdSet, objNameList = AllNodes(data, i)

        # 이름이 중복되면 value 값 갱신됨
        # 이름 하나에 하나의 i 값만 갖는 dict
        # idNameDict = {ObjIdx : SynsetName} 를 만든기 위해 전체 objIdList를 사용

        attrNameList = []  # name attr  추가를 위한 코드
        for kId in objIdSet:
            attrNameList.append(synsDict[str(kId)])
        # idNameDict = {이미지 내 ObjIdx : NameList}
        idNameDict = {str(idx): synsetName for idx, synsetName in zip(
            objIdSet, attrNameList)}  # name attr  추가를 위한 코드

        newObjName = []  # name attr  추가를 위한 코드
        newSubjName = []  # name attr  추가를 위한 코드

        # Obj,Subj의 Id에 해당하는 SynsetName List 생성
        for oId in objId:
            try:
                newObjName.append(idNameDict[str(oId)])
            except:
                newObjName.append('')

        for sId in subjId:
            try:
                newSubjName.append(idNameDict[str(sId)])
            except:
                newSubjName.append('')

        # 자기 자신에 참조하는 node가 있는 Idx List 생성
        recurRowId = []
        for j in range(len(objId)):
            if objId[j] == subjId[j]:
                recurRowId.append(j)

        df_edge = pd.DataFrame(
            {"objId": objId, "subjId": subjId, "newObjName": newObjName, "newSubjName": newSubjName, })

        # 자기 자신을 참조하는 노드의 relationship 삭제
        if recurRowId != 0:
            for idx in recurRowId:
                df_edge = df_edge.drop(index=idx)

        gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')

        # --------- ^^^ Graph 생성, graph에 name, Origin ObjId를 attribute로 추가함 ^^^ ------------------
        #                   자기 자신 참조하는 중복 제거, synset name 적용

        # ----------- vvv 이웃 노드에 동일한 이름을 가진 노드가 5개 이상인 경우 동일 id로 변환 vvv -------------

        nodesList = sorted(list(gI.nodes))
        objIdSet = df_edge['objId'].tolist() + df_edge['subjId'].tolist()
        objNameList = df_edge['newObjName'].tolist() + \
            df_edge['newSubjName'].tolist()

        # 위에서 제거된 값을 위해 변경
        objId = df_edge['objId'].tolist()
        subjId = df_edge['subjId'].tolist()

        neighUpp5 = []  # neighbors 5개 이상인 것들의 nodeId
        for nodeId in nodesList:
            if (len(list(gI.neighbors(nodeId)))) >= 5:
                neighUpp5.append(nodeId)

        '''
        Neighbors의 objectName 확인, 5개 이상 동일한 경우, 해당 Neighbor의 Id를 묶고, sort 함.
        이후 전체 ObjId List에서 바꿔줌. Id로 이름 호출. get_key 사용해서 이름으로 Id 호출
        '''

        # 전체 노드 id에 대해 변경해야 할 Id List fId = 리스트들에서 제일 작은 Id, totalList = [[아이디들] , []],nameList = [동일한 ObjName] <- 예외처리를 위해
        fId = []
        totalList = []

        for nodeId in neighUpp5:
            neighbors = list(gI.neighbors(nodeId))
            neiNames = [idNameDict[str(k)] for k in neighbors]

            sameName = list(Counter(neiNames).keys())
            sameNums = list(Counter(neiNames).values())
            sameUpp5Idx = []
            for idx in range(len(sameNums)):
                if sameNums[idx] >= 5:
                    sameUpp5Idx.append(idx)

            # delTargetnames : 한 노드의 이웃노드면서 5개 이상 중복되는 objectName List
            delTargetNames = []

            # todo 분기처리 - 예외 단어 추가하기 / 예외단어 : 중복이어도 허용하는 important Name
            exceptionalWords = []
            if sameUpp5Idx != 0:
                for idx in sameUpp5Idx:
                    # todo 분기처리 - 여기서 고려대상 나오면 걍 넘기기
                    if sameName[idx] in exceptionalWords:
                        continue
                    else:
                        delTargetNames.append(sameName[idx])

            if len(delTargetNames) != 0:
                delTargetIds = []
                for name in delTargetNames:
                    for key, value in idNameDict.items():
                        if name == value:
                            delTargetIds.append(key)

                delTargetIds = sorted(delTargetIds)
                fId.append(delTargetIds[0])
                totalList.append(delTargetIds)

        # todo 앞서 변경된 Id가 나중에 변경된 Id 값에 의해 왕창 늘어날 가능성 고려 및 코드 수정 필요
        # replaceDict = {delTargetIds : delTargetIds 중 변경 대상이 되는 Id}
        replaceDict = {}
        for idx in range(len(totalList)):
            for jIdx in range(len(totalList[idx])):
                replaceDict[str(totalList[idx][jIdx])] = fId[idx]

        newObjList = []
        newSubjList = []
        if (len(replaceDict) != 0):
            for idx in objId:
                try:
                    newObjList.append(replaceDict[str(idx)])  # 교체 대상인 경우 교체
                except KeyError:
                    # 이웃 노드에서 중복이 아니어서 교체 대상이 아닌 경우 기존 Id
                    newObjList.append(str(idx))

            for idx in subjId:
                try:
                    newSubjList.append(replaceDict[str(idx)])  # 교체 대상인 경우 교체
                except KeyError:
                    # 이웃 노드에서 중복이 아니어서 교체 대상이 아닌 경우 기존 Id
                    newSubjList.append(str(idx))

        if len(newObjList) != 0:
            newObjId, newSubjId = newObjList, newSubjList
        else:
            newObjId, newSubjId = objId, subjId
        # id 값으로 name 호출 - obj / subj -> idtoName
        newObjName = [idNameDict[str(idx)] for idx in newObjId]
        newSubjName = [idNameDict[str(idx)] for idx in newSubjId]

        df_new = pd.DataFrame({"objId": newObjId, "subjId": newSubjId,
                               "objName": newObjName, "subjName": newSubjName})

        df_new['objId'] = df_new['objId'].astype(int)
        df_new['subjId'] = df_new['subjId'].astype(int)

        objIdSet = df_new['objId'].tolist() + df_new['subjId'].tolist()
        #objIdSet = list(set(df_new['objId'].tolist() + df_new['subjId'].tolist()))
        #objIdSet = gI.nodes()

    # objNameList = df_new['objName'].tolist() + df_new['subjName'].tolist()

        # totalEmbDict = {Name : Embedding}
        # embDict = { 해당 이미지 내 objId : textEmbedding 값}
        # embList = [totalEmbDict[synsDict[str(idx)]] for idx in objIdSet]
        # embDict = {idx: emb for idx, emb in zip(objIdSet, embList)}

        gI = nx.from_pandas_edgelist(df_new, source='objId', target='subjId')
        for index, row in df_new.iterrows():
            gI.nodes[row['objId']]['name'] = row["objName"]  # name attr
            gI.nodes[row['subjId']]['name'] = row['subjName']  # name attr

            gI.nodes[row['objId']]['originId'] = row['objId']  # originId attr
            # originId attr
            gI.nodes[row['subjId']]['originId'] = row['subjId']

        nodesList = sorted(list(gI.nodes))
        embList = [totalEmbDict[synsDict[str(idx)]] for idx in nodesList]
        embDict = {idx: emb for idx, emb in zip(nodesList, embList)}

        for idx in range(len(nodesList)):  # nodeId
            nodeId = nodesList[idx]
            emb = embDict[nodeId]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
            for j in range(3):  # Embedding 값은 [3,]인데, 원소 각각을 특징으로 node에 할당
                nx.set_node_attributes(gI, {nodeId: emb[j]}, "f" + str(j))
        # node relabel - graph에서 노드 id 0부터 시작하도록 ---
        dictIdx = {nodeId: idx for idx, nodeId in enumerate(nodesList)}
        gI = nx.relabel_nodes(gI, dictIdx)
        gList.append(gI)

    # < node[nId]['attr'] = array(float)
    # with open("data/networkx_ver2.pickle", "wb") as fw:
    #     pickle.dump(gList, fw)

    # with open("data/networkx_ver2.pickle", "rb") as fr:
    #     data = pickle.load(fr)

    gId = 0
    gI = gList[gId]
    # print(data)
    print('data[gId] : ', gList[gId])
    print('data[gId].node : ', gList[gId].nodes(data=True))

    print(list(synsDict.values())[0])
    print('synsDict len : ', len(list(set(synsDict.values()))))

    plt.figure(figsize=[15, 7])
    nx.draw(gI, with_labels=True)
    plt.show()


''' Image에 해당하는 모든 ObjectId - ObjectName '''


def AllNodes(data, imgId):
    objectIds = []
    objectNames = []
    objects = data[imgId]["objects"]
    for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
        objectIds.append(objects[j]['object_id'])
        objectNames.append(objects[j]['names'][0])

    return objectIds, objectNames


''' Relationship로 ObjId-SubjId 간 Edge 추출'''


def AllEdges(data, ImgId):

    objId = []
    subjId = []
    relatiohship = []  # 혹시 모르니까 추가할래 Id 말고 Predicate로~
    edgeId = []  # csv에 저장할 때 한 번에 얹으려고.. 0번부터 시작
    weight = []  # default 1
    # relationships에 따라 중복 허용하고 list에 append하므로 df로 출력하면 엣지를 확인 할 수 있음
    # 이를 이용해 Adj나 Featurematrix의 row 순서를 맞추려면 중복 제거 / 중복 제거 후 순서맞춰서 embedding 값 추가해야함

    imageDescriptions = data[ImgId]["relationships"]

    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
        objId.append(imageDescriptions[j]['object_id'])
        subjId.append(imageDescriptions[j]['subject_id'])
        relatiohship.append(imageDescriptions[j]['predicate'])
        edgeId.append(j)
        weight.append(1)

    return objId, subjId, relatiohship, edgeId, weight


def FeatEmbeddPerTotal_model(objectNames):
    # a = []
    # a.append(objectNames)
    # model = FastText(a, vector_size=10, workers=4, sg=1, word_ngrams=1)
    model = FastText()
    # vocab가 너무 적은 경우(5개 정도로) 정상 작동하지 않아 id 242의 objectName을 임의로 삽입함 -> Top Object를 넣는 게 더 나을 것 같기두..
    model.build_vocab(objectNames)
    model = FastText(objectNames, vector_size=3,
                     workers=4, sg=1, word_ngrams=1)
    model.build_vocab(objectNames)
    embedding = []
    for i in objectNames:  # objName 개수만큼만 반복하니까 vocab에 추가해 준 거 신경 X. Id:Embedding 값으로 dict 생성
        embedding.append(model.wv[i])
    # objectNames, Embedding 형태로 Dict 저장
    totalEmbDict = {name: value for name, value in zip(objectNames, embedding)}
    #embDict = {name: torch.FloatTensor(value) for name, value in zip(objectIds, embedding)}

    return model, totalEmbDict


if __name__ == "__main__":
    train_node = True
    make_graph(train_node)
