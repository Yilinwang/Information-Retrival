import pickle
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from collections import defaultdict
from math import log

def cal_tfidf(model_dir):
    idf = dict()
    tfidf = defaultdict(dict)
    n = 46972
    cur_idf = -1
    cur_term = -1
    cur_docs = list()
    with open(model_dir+'/inverted-index') as fp:
        for line in fp:
            tmp = line.strip().split()
            if len(tmp) == 3:
                idf[cur_term] = cur_idf
                for doc, count in cur_docs:
                    tfidf[doc][cur_term] = int(count) * cur_idf
                cur_idf = log(n/int(tmp[2]))
                cur_term = tmp[0] if tmp[1] == '-1' else tmp[0]+':'+tmp[1]
                cur_docs.clear()
            elif len(tmp) == 2:
                cur_docs.append((int(tmp[0]), int(tmp[1])))
    return tfidf

def parse(query_file):
    root = ET.parse(query_file).getroot()

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-r', action='store_true')
    parser.add_argument('-i', '--query_file', type=str)
    parser.add_argument('-o', '--ranked_list', type=str)
    parser.add_argument('-m', '--model_dir', type=str)
    parser.add_argument('-d', '--ntcir_dir', type=str)
    args = parser.parse_args()
    if(args.r):
        pass
    return args

def main():
    args = get_args()
    #query = parse(args.query_file)
    tfidf = cal_tfidf(args.model_dir)
    with open('basic_tfidf', 'wb') as fp:
        pickle.dump(tfidf, fp)

if __name__ == '__main__':
    main()
