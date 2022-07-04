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

    def get_key(dict, val):
        for key, value in dict.items():
            if val == value:
                return key
        return "key doesn't exist"

    '''
        2. noun이 아닐 때 NLTK를 통해서 명사를 찾아 name으로 사용 가능한(synset으로 대체 가능한) noun 추출
            -> noun이 두 개 일 경우 기존 synset List에서 가장 많이 사용되는 단어를 사용하고, 개수가 일치 할 경우 [0]의 단어를 name으로 함
            -> NLTK로 분할한 noun들이 기존 synset이 아닌 경우 전체를 통째로 synset으로 만들기
    '''

    def extractNoun(noun, synsDict, synNameCnter):
        conllDict = {word: tag for word,
                     tag in conll2000.tagged_words(tagset='universal')}

        words = noun.split(' ')
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
        if len(nouns) != 0:
            for i in nouns:
                try:
                    nInSynsDictList.append(synsDict[i])
                except:
                    continue

        # synset에 해당되는 noun 중 언급이 많은 단어 선택
        name = ''
        cnt = 0
        if len(nInSynsDictList) != 0:
            for i in nInSynsDictList:
                if cnt < synNameCnter[i]:
                    name = i
                    cnt = synNameCnter[i]
        else:
            name = "_".join(sorted(nouns))
        return name

    gList = []
    imgCnt = 1000
    start = time.time()
    with open('common/data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
    end = time.time()
    # 파일 읽는데 걸리는 시간 : 24.51298 sec
    print(f"파일 읽는데 걸리는 시간 : {end - start:.5f} sec")

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
        1. originDict{synset을 갖는 objId : objName}에서 nonSynsetName의 원소가 있는 경우, objId로 synsDict에서 synsetName을 찾음
        2. nonSynsetName/Id에서 해당 원소를 제외하고, synsDict에 추가함(objId : objName(synset))

        없는 경우, 2로 넘어감
    '''

    synsDict = {idx: name for idx, name in zip(originIdList, synsetList)}
    nonSynsDict = {name: value for name,
                   value in zip(nonSysnIdList, nonSysnNameList)}

    for i in range(len(nonSysnNameList)):
        try:
            # 1. originDict{synset을 갖는 objId : objName}에서 nonSynsetName의 원소가 있는 경우,
            sameNameId = get_key(originDict, nonSysnNameList[i])
            synsDict[nonSysnIdList[i]] = synsDict[sameNameId]

        except:
            # todo 2. noun이 아닐 때 NLTK를 통해서 명사를 찾아 name으로 사용 가능한(synset으로 대체 가능한) noun 추출
            name = extractNoun(nonSysnNameList[i], synsDict, synNameCnter)
            synsDict[nonSysnIdList[i]] = name

    objNamesList = []
    newObjIdListforSynset = []
    for imgId in tqdm(range(imgCnt)):
        objectIds, objectNames = AllNodes(data, imgId)
        for id in objectIds:
            try:
                objNamesList.append(synsDict[str(id)])  # synset Id로 변경
                # synset 이 없어 Dict에 없는 ObjId는 제외함
                newObjIdListforSynset.append(id)
            except:
                continue
    totalEmbDict = FeatEmbeddPerTotal(list(set(objNamesList)))

    for i in tqdm(range(imgCnt)):
        objId, subjId, relatiohship, edgeId, weight = AllEdges(data, i)
        # networkX graph 객체 생성 ---
        objIdSet, objNameList = AllNodes(data, i)

        objIdList = []
        objNameListSynset = []
        for id in objIdSet:
            try:
                objNameListSynset.append(synsDict[str(id)])
                objIdList.append(id)
            except:
                continue
        # if (i == 5):
        #     print('objNameListSynset : ', objNameListSynset)
        #     print('objIdList : ', objIdList)

        '''
            Object Name List를 기준으로 ReLabeling(String -> Int)
            Node List 생성
            1. ObjNameList의
                relabelDict - {name : newIdx}
            2. newObjNameList = []
            2. idIdxDict = {ObjIdSet : relabelDict[objIdSet]}
            3. newObjId = idIdxDict[objId(i)]
            newSubjId = idIdxDict[subjId(i)]

            이름에 대한 NodeList
        '''
        # 이름이 중복되면 value 값 갱신됨
        # 이름 하나에 하나의 i 값만 갖는 dict
        relabelDict = {objName: i for i, objName in zip(objIdList,
                                                        objNameListSynset)}  # relabel을 위한 objId
        newObjIdList = []
        attrNameList = []  # name attr  추가를 위한 코드
        for kId in objIdList:
            attrNameList.append(synsDict[str(kId)])
            # name attr  추가를 위한 코드
            newObjIdList.append(relabelDict[synsDict[str(kId)]])

        idIdxDict = {name: value for name,
                     value in zip(objIdList, newObjIdList)}

        idNameDict = {name: value for name, value in zip(
            objIdList, attrNameList)}  # name attr  추가를 위한 코드

        newObjId = []
        newSubjId = []

        oriObjId = []
        oriSubjId = []

        newObjName = []  # name attr  추가를 위한 코드
        newSubjName = []  # name attr  추가를 위한 코드

        # 자기 자신에 참조하는 node 삭제
        for oId in objId:
            try:
                oriObjId.append(oId)
                newObjId.append(idIdxDict[oId])
                newObjName.append(idNameDict[oId])
            except:
                newObjId.append('')
                newObjName.append('')
                oriObjId.append('')

        for sId in subjId:
            try:
                oriSubjId.append(sId)
                newSubjId.append(idIdxDict[sId])
                newSubjName.append(idNameDict[sId])
            except:
                newSubjId.append('')
                newSubjName.append('')
                oriSubjId.append('')

        recurRowId = []
        for j in range(len(newObjId)):
            if newObjId[j] == newSubjId[j]:
                recurRowId.append(j)

        df_edge = pd.DataFrame({"objId": newObjId, "subjId": newSubjId, "oriObjId": oriObjId, "oriSubjId": oriSubjId,
                                "newObjName": newObjName, "newSubjName": newSubjName, })

        # todo 자기 자신을 참조하는 노드의 relationship 삭제
        if recurRowId != 0:
            for idx in recurRowId:
                df_edge = df_edge.drop(index=idx)
                # df_name = df_name.drop(index=idx)

        # todo synset에 없어 ''로 append 된 경우 dropna로 해당 행 삭제(해당 relationship 삭제)
        # df_edge = df_edge.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
        df_edge = df_edge.replace(r'', np.nan, regex=True)
        df_edge = df_edge.dropna()

        df_edge['objId'] = df_edge['objId'].astype(int)
        df_edge['subjId'] = df_edge['subjId'].astype(int)

        # df_edge = pd.DataFrame({"objId": objId, "subjId": subjId, })
        gI = nx.from_pandas_edgelist(df_edge, source='objId', target='subjId')

        # print("gI.nodes : ", gI.nodes)
        nodesList = sorted(list(gI.nodes))
        # print("nodesList : ", nodesList)
        # print("len(nodesList) : ", len(nodesList))

        # embDict = ut.MatchDictImage(newObjId+newSubjId,  newObjName+newSubjName, totalEmbDict)
        # node attribute 부여 ---
        objIdSet = df_edge['objId'].tolist() + df_edge['subjId'].tolist()
        objNameList = df_edge['newObjName'].tolist(
        ) + df_edge['newSubjName'].tolist()  # todo
        embDict = MatchDictImage(objIdSet, objNameList, totalEmbDict)

        neighUpp5 = []
        for nodeId in nodesList:  # nodeId
            # neighbors = gI.neighbors(nodeId)
            if (len(list(gI.neighbors(nodeId)))) >= 5:
                neighUpp5.append(nodeId)
            # neighbors 5개 이상인 것들의 nodeId

        idToName = {id: name for id, name in zip(objIdSet, objNameList)}

        idToOriginId = {id: name for id, name in zip(objIdSet, objNameList)}

        '''
        Neighbors의 objectName 확인, 5개 이상 동일한 경우, 해당 Neighbor의 Id를 묶고, sort 함.
        이후 전체 ObjId List에서 바꿔줌. Id로 이름 호출. get_key 사용해서 이름으로 Id 호출
        '''

        # 전체 노드 id에 대해 변경해야 할 Id List fId = 리스트들에서 제일 작은 Id, totalList = [[아이디들] , []],nameList = [동일한 ObjName] <- 예외처리를 위해
        fId = []
        totalList = []
        nameList = []

        for nodeId in neighUpp5:
            neighbors = list(gI.neighbors(nodeId))
            neiNames = [idToName[k] for k in neighbors]

            sameName = list(Counter(neiNames).keys())
            sameNums = list(Counter(neiNames).values())
            sameUpp5 = list(filter(lambda num: num >= 5, sameNums))

            # todo 분기처리 - 예외 단어 추가하기
            exceptionalWords = []

            a = []
            if sameUpp5 != 0:
                for i in sameUpp5:
                    # todo 분기처리 - 여기서 고려대상 나오면 걍 넘기기
                    if sameName[sameNums.index(i)] in exceptionalWords:
                        continue
                    else:
                        a.append(sameName[sameNums.index(i)])
            if len(a) != 0:
                b = []
                for name in a:
                    for key, value in idToName.items():
                        if name == value:
                            b.append(key)
                if len(b) != 0:
                    b = sorted(b)
                    fId.append(b[0])
                    totalList.append(b)
                    nameList += a

        # 동일한 이름이 있을 때 nameList의 원소가 동일한 것들의 nameList.index() 구해서,
        # totalList(idx) 끼리 더하고, sort 해서 fId
        # if len(fId) != len(nameList) :
        # 하나에 너무 많이 겹치는 거 아닌가 그러면?
        # dictionary = totalList[i][j] : fId[i]

        replaceDict = {}
        for i in range(len(totalList)):
            for j in range(len(totalList[i])):
                replaceDict[str(totalList[i][j])] = fId[i]

        # print('replace : ', replaceDict)

        newObjList = []
        newSubjList = []
        if (len(replaceDict) != 0):
            for i in objId:
                try:
                    newObjList.append(replaceDict[str(i)])
                    # print('replaceDict[str(i)] : ', replaceDict[str(i)])
                    # print(i, '를 ', replaceDict[str(i)], '로')
                except KeyError:
                    newObjList.append(i)

            for i in subjId:
                try:
                    newSubjList.append(replaceDict[str(i)])
                except KeyError:
                    newSubjList.append(i)

            newObjId, newSubjId = newObjList, newSubjList
            df_new = pd.DataFrame({"objId": newObjId, "subjId": newSubjId, })
            gI = nx.from_pandas_edgelist(
                df_new, source='objId', target='subjId')
            nodesList = sorted(list(gI.nodes))

        for i in range(len(nodesList)):  # nodeId
            nodeId = nodesList[i]
            emb = embDict[nodeId]  # nodeId로 그래프 내 embDict(Id-Emb)에서 호출
            for j in range(3):  # Embedding 값은 [3,]인데, 각 원소를 특징으로 node에 할당
                nx.set_node_attributes(
                    gI, {nodeId: float(emb[j])}, "f" + str(j))

        for index, row in df_edge.iterrows():
            gI.nodes[row['objId']]['name'] = row['newObjName']
            gI.nodes[row['subjId']]['name'] = row['newSubjName']
            gI.nodes[row['objId']]['originId'] = row['objId']
            gI.nodes[row['subjId']]['originId'] = row['objId']

    # graph에서 노드 id 0부터 시작하도록 ---
        listA = list(set(newObjId + newSubjId))
        listIdx = range(len(listA))
        dictIdx = {name: value for name, value in zip(listA, listIdx)}
        gI = nx.relabel_nodes(gI, dictIdx)
        gList.append(gI)

    # < node[nId]['attr'] = array(float)
    # with open("data/networkx_ver1.pickle", "wb") as fw:
    #     pickle.dump(gList, fw)
    #
    # with open("data/networkx_ver1.pickle", "rb") as fr:
    #     data = pickle.load(fr)

    gId = 0
    gI = gList[gId]
    # print(data)
    # print('data[gId] : ', gList[gId])
    # print('data[gId].node : ', gList[gId].nodes(data=True))

    # print(list(synsDict.values())[0])
    # print('synsDict len : ', len(list(set(synsDict.values()))))

    plt.figure(figsize=[15, 7])
    nx.draw(gI, with_labels=True)
    plt.show()


def blank_nan(x):
    if x == '':
        x = np.nan
    return x


''' Image에 해당하는 모든 ObjectId - ObjectName '''
# Object.json


def AllNodes(data, imgId):
    objectIds = []
    objectNames = []
    objects = data[imgId]["objects"]
    for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
        objectIds.append(objects[j]['object_id'])
        objectNames.append(objects[j]['names'][0])

    return objectIds, objectNames


# SceneGraph.json 사용
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


'''
    전체 word에 대한 fasttext Embedding값  - 01
'''


def FeatEmbeddPerTotal(objectNames):
    # a = []
    # a.append(objectNames)
    # model = FastText(a, vector_size=10, workers=4, sg=1, word_ngrams=1)
    model = FastText()
    # vocab가 너무 적은 경우(5개 정도로) 정상 작동하지 않아 id 242의 objectName을 임의로 삽입함 -> Top Object를 넣는 게 더 나을 것 같기두..
    model.build_vocab(objectNames)
    model = FastText(objectNames, vector_size=3,
                     workers=4, sg=1, word_ngrams=1)
    # model.build_vocab(objectNames)
    embedding = []
    for i in objectNames:  # objName 개수만큼만 반복하니까 vocab에 추가해 준 거 신경 X. Id:Embedding 값으로 dict 생성
        embedding.append(model.wv[i])
    # objectNames, Embedding 형태로 Dict 저장

    totalEmbDict = {name: value
                    for name, value in zip(objectNames, embedding)}
    # embDict = {name: torch.FloatTensor(value) for name, value in zip(objectIds, embedding)}

    return totalEmbDict


'''
    전체 word에 대한 fasttext Embedding값  - 02 < 각 이미지 별 node Id/Name 매칭 
    Dict(wordEmbedding - Word:Embedding) 을 이용해 Id : Embedding Dict를 반환하도록 변경    
'''


def MatchDictImage(objectIds, objectNames, totalEmbDict):
    embList = []
    for i in objectNames:  # objName 개수만큼만 반복하니까 vocab에 추가해 준 거 신경 X. Id:Embedding 값으로 dict 생성
        embList.append(totalEmbDict[i])
    # objectNames, Embedding 형태로 Dict 저장
    embDict = {name: value for name, value in zip(objectIds, embList)}

    return embDict


if __name__ == "__main__":
    train_node = True
    make_graph(train_node)
