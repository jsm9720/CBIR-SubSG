'''
만든이 : SeongMo Jeong
개요 : 파일로 저장되어있는 이미지 결과를 plot 해주는 코드
'''
from common.visualizerNetx import plotOnImg
import pickle
import os


def main():

    for path in os.listdir('plots/data'):
        filename = path.split(".")
        os.mkdir('plots/'+filename[0])
        with open("plots/data/"+path, "rb") as fr:
            results = pickle.load(fr)

        cnt = 0
        sub_num = 0
        for result in results:
            for i, G in result:
                plotOnImg(i, G, "plots/" +
                          filename[0]+"/"+str(sub_num)+"_"+str(cnt))
                cnt += 1
            cnt = 0
            sub_num += 1


if __name__ == "__main__":
    main()
