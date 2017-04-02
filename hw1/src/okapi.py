import pickle
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from collections import defaultdict
from math import log
from math import sqrt


def read_model(model_dir):
    with open(model_dir+'/file-list') as fp:
        l = [line.strip().split('/')[-1].lower() for line in fp]
    file_list = dict(zip(range(len(l)), l))
    with open(model_dir+'/vocab.all') as fp:
        encode = fp.readline()
        vl = [line.strip() for line in fp]
    vocab = dict(zip(vl, range(1, len(vl))))
    return file_list, vocab

def dd():
    return defaultdict(int)

def cal_tf_idf(model_dir, N):
    idf = defaultdict(int)
    tf = defaultdict(dd)
    doc_len = defaultdict(int)
    cur_term = -1
    with open(model_dir+'/inverted-index') as fp:
    #with open('small') as fp:
        for line in fp:
            tmp = line.strip().split()
            if len(tmp) == 3:
                cur_term = tmp[0] if tmp[1] == '-1' else tmp[0]+':'+tmp[1]
                df = int(tmp[2])
                idf[cur_term] = log((N-df+0.5)/df+0.5)
            elif len(tmp) == 2:
                tf[int(tmp[0])][cur_term] = int(tmp[1])
                #if ':' not in cur_term:
                doc_len[int(tmp[0])] += int(tmp[1])
    ave = sum(doc_len.values()) / N
    return tf, idf, doc_len, ave

def parse(query_file, vocab):
    query = dict()
    root = ET.parse(query_file).getroot()
    for topic in root.findall('topic'):
        num = topic.find('number').text.strip()[-3:]
        d = defaultdict(int)
        for term in bigram(topic.find('concepts').text.strip('。\n ').split('、'), vocab):
            d[term] += 1
        for term in bigram(topic.find('title').text.strip('。\n ').split('、'), vocab):
            d[term] += 1
        for term in bigram(topic.find('question').text.strip('。\n ').split('、'), vocab):
            d[term] += 1
        for term in bigram(topic.find('narrative').text.strip('。\n ').split('、'), vocab):
            d[term] += 1
        query[num] = d
    return query

def bigram(q, vocab):
    l = list()
    for w in q:
        w = [str(vocab[x]) for x in w if x in vocab]
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
    parser.add_argument('-k1', type=float)
    parser.add_argument('-k3', type=float)
    parser.add_argument('-b', type=float)
    args = parser.parse_args()
    if(args.r):
        pass
    return args

def main():
    args = get_args()
    file_list, vocab = read_model(args.model_dir)
    tf, idf, doc_len, ave = cal_tf_idf(args.model_dir, len(file_list))
    query = parse(args.query_file, vocab)

    b = args.b
    k1 = args.k1
    k3 = args.k3

    doc_vec = defaultdict(lambda: defaultdict(int))
    doc_sum = defaultdict(int)
    for doc in tf:
        for term in tf[doc]:
            doc_vec[doc][term] = ((k1+1)*tf[doc][term]/(k1*((1-b)+b*(doc_len[doc]/ave))+tf[doc][term])) * idf[term]
            doc_sum[doc] += doc_vec[doc][term]**2
        doc_sum[doc] = sqrt(doc_sum[doc])
    query_vec = defaultdict(lambda: defaultdict(int))
    query_sum = defaultdict(int)
    for q in query:
        for term in query[q]:
            query_vec[q][term] = (((k3+1)*query[q][term])/(k3+query[q][term])) * idf[term]
            query_sum[q] += query_vec[q][term]**2
        query_sum[q] = sqrt(query_sum[q])

    result = defaultdict(lambda: defaultdict(int))
    for doc in tf:
        for q in query:
            for term in query[q]:
                if term in idf:
                    result[q][doc] += query_vec[q][term] * doc_vec[doc][term]
            result[q][doc] /= query_sum[q] * doc_sum[doc]


    print('query_id,retrieved_docs')
    for qnum in result:
        print(qnum, end=',')
        print(' '.join([file_list[x[0]] for x in sorted(result[qnum].items(), key=lambda x: x[1], reverse=True)[:100]]))


if __name__ == '__main__':
    main()
