import pickle
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from collections import defaultdict
from math import log


def read_model(model_dir):
    with open(model_dir+'/file-list') as fp:
        l = [line.strip().split('/')[-1].lower() for line in fp]
    file_list = dict(zip(range(len(l)), l))
    with open(model_dir+'/vocab.all') as fp:
        encode = fp.readline()
        vl = [line.strip() for line in fp]
    vocab = dict(zip(vl, range(1, len(vl))))
    return file_list, vocab

def cal_tfidf(model_dir, N):
    idf = dict()
    tf = defaultdict(lambda: defaultdict(int))
    cur_idf = -1
    cur_term = -1
    cur_docs = list()
    max_tf = -1
    with open(model_dir+'/inverted-index') as fp:
        for line in fp:
            tmp = line.strip().split()
            if len(tmp) == 3:
                idf[cur_term] = cur_idf
                for doc, count in cur_docs:
                    tf[doc][cur_term] = int(count)
                    if int(count) > max_tf:
                        max_tf = int(count)
                cur_idf = log(N/int(tmp[2]))
                cur_term = tmp[0] if tmp[1] == '-1' else tmp[0]+':'+tmp[1]
                cur_docs.clear()
            elif len(tmp) == 2:
                cur_docs.append((int(tmp[0]), int(tmp[1])))
    for doc in tf:
        for term in tf[doc]:
            tf[doc][term] = (tf[doc][term] / max_tf)
    return tf, idf

def parse(query_file, idf, vocab):
    query = list()
    root = ET.parse(query_file).getroot()
    for topic in root.findall('topic'):
        num = topic.find('number').text.strip()[-3:]
        d = defaultdict(int)
        max_tf = -1
        for term in bigram(topic.find('concepts').text.strip('。\n ').split('、'), vocab):
            d[term] += 1
            if d[term] > max_tf:
                max_tf = d[term]
        for term in d:
            d[term] = d[term]/max_tf
        query.append((num, d))
    return query

def bigram(q, vocab):
    l = list()
    for w in q:
        w = [str(vocab[x]) for x in w]
        for i in range(len(w)-1):
            l.append(w[i])
            l.append(':'.join(w[i:i+2]))
        l.append(w[-1])
    return l

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
    file_list, vocab = read_model(args.model_dir)
    tf, idf = cal_tfidf(args.model_dir, len(file_list))
    query = parse(args.query_file, idf, vocab)

    result = defaultdict(lambda: defaultdict(int))
    for doc in tf:
        for qnum, q in query:
            for term in idf:
                result[qnum][doc] += ((tf[doc][term]*0.5+0.5)*idf[term])*((q[term]*0.5+0.5)*idf[term])
    print('query_id,retrieved_docs')
    for qnum in result:
        print(qnum, end=',')
        print(' '.join([file_list[x[0]] for x in sorted(result[qnum].items(), key=lambda x: x[1], reverse=True)[:100]]))


if __name__ == '__main__':
    main()
