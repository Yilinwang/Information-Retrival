import pickle
import jieba
import re
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from collections import defaultdict
from collections import Counter
from math import log
from math import sqrt


def read_model(model_dir):
    with open(model_dir+'/file-list') as fp:
        l = [line.strip().split('/')[-1].lower() for line in fp]
    file_list = dict(zip(range(len(l)), l))
    with open(model_dir+'/vocab.all') as fp:
        encode = fp.readline()
        vl = [line.strip() for line in fp]
    vocab = dict(zip(vl, range(1, len(vl)+1)))
    return file_list, vocab

def cal_tf_idf(model_dir, N):
    idf = defaultdict(int)
    tf = defaultdict(lambda: defaultdict(dict))
    doc_len = defaultdict(int)
    cur_term = -1
    with open(model_dir+'/inverted-index') as fp:
    #with open('small') as fp:
        for line in fp:
            tmp = line.strip().split()
            if len(tmp) == 3:
                cur_term = tmp[0] if tmp[1] == '-1' else tmp[0]+':'+tmp[1]
                df = int(tmp[2])
                idf[cur_term] = log((N-df+0.5)/(df+0.5))
            elif len(tmp) == 2:
                tf[int(tmp[0])][cur_term] = int(tmp[1])
                doc_len[int(tmp[0])] += int(tmp[1])
    ave = sum(doc_len.values()) / N
    return tf, idf, doc_len, ave

def parse(query_file, vocab):
    query = dict()
    root = ET.parse(query_file).getroot()
    for topic in root.findall('topic'):
        num = topic.find('number').text.strip()[-3:]
        c = Counter()
        c.update(list(jieba.cut_for_search(topic.find('concepts').text.strip('。\n '))))
        c.update(list(jieba.cut_for_search(topic.find('title').text.strip('。\n '))))
        c.update(list(jieba.cut_for_search(topic.find('question').text.strip('。\n '))))
        c.update(list(jieba.cut_for_search(topic.find('narrative').text.strip('。\n '))))
        query[num] = dict(c)
    return query

def bigram(q, vocab):
    l = list()
    for w in q:
        if not len(w) == 0:
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
    parser.add_argument('-doc', type=int)
    args = parser.parse_args()
    return args

def cal_result(query_vec, query_sum, doc_vec, doc_sum):
    result = defaultdict(lambda: defaultdict(int))
    for doc in doc_vec:
        for q in query_vec:
            for term in query_vec[q]:
                if term in doc_vec[doc]:
                    result[q][doc] += query_vec[q][term] * doc_vec[doc][term]
            #result[q][doc] /= query_sum[q] * doc_sum[doc]
    return result

def preprocess(model_dir):
    tf = dict()
    df = defaultdict(int)
    doc_len = defaultdict(int)
    with open(model_dir+'/file-list') as listfp:
        for line in listfp:
            root = ET.parse(line.strip()).getroot()
            doc_id = root.find('doc').find('id').text.strip()
            c = Counter()
            c.update(list(jieba.cut_for_search(root.find('doc').find('title').text.strip())))
            for p in root.find('doc').find('text').findall('p'):
                c.update(list(jieba.cut_for_search(p.text.strip())))
            for term in c.keys():
                df[term] += 1
            doc_len[doc_id] = sum(c.values())
            tf[doc_id] = dict(c)
    ave = sum(doc_len.values())/len(doc_len)
    return tf, df, doc_len, ave

def main():
    args = get_args()
    #file_list, vocab = read_model(args.model_dir)
    #tf, idf, doc_len, ave = cal_tf_idf(args.model_dir, len(file_list))
    tf, df, doc_len, ave = preprocess(args.model_dir)
    query = parse(args.query_file, vocab)

    b = 0.85
    k1 = 1.4
    k3 = 350

    N = len(doc_len)
    doc_vec = defaultdict(lambda: defaultdict(int))
    doc_sum = defaultdict(int)
    for doc in tf:
        for term in tf[doc]:
            idf = log((N-df[term]+0.5)/(df[term]+0.5))
            doc_vec[doc][term] = (((k1+1)*tf[doc][term])/(k1*((1-b)+(b*(doc_len[doc]/ave)))+tf[doc][term])) * idf
            doc_sum[doc] += doc_vec[doc][term]**2
        doc_sum[doc] = sqrt(doc_sum[doc])
    query_vec = defaultdict(lambda: defaultdict(int))
    query_sum = defaultdict(int)
    for q in query:
        for term in query[q]:
            query_vec[q][term] = (((k3+1)*query[q][term])/(k3+query[q][term]))
            query_sum[q] += query_vec[q][term]**2
        query_sum[q] = sqrt(query_sum[q])

    result = cal_result(query_vec, query_sum, doc_vec, doc_sum)

    if args.r:
        b = 0.8
        d = 10
        for q in query:
            rel_doc = list(zip(*sorted(result[q].items(), key=lambda x: x[1], reverse=True)))[0][:d]
            for doc in rel_doc:
                for term in doc_vec[doc]:
                    query_vec[q][term] += (b/doc) * doc_vec[doc][term]
            query_sum[q] = 0
            for term in query_vec[q]:
                query_sum[q] += query_vec[q][term]**2
            query_sum[q] = sqrt(query_sum[q])
        result = cal_result(query_vec, query_sum, doc_vec, doc_sum)

    with open(args.ranked_list, 'w') as fp:
        fp.write('query_id,retrieved_docs\n')
        for qnum in result:
            fp.write(str(qnum)+',')
            fp.write(' '.join([file_list[x[0]] for x in sorted(result[qnum].items(), key=lambda x: x[1], reverse=True)[:100]])+'\n')


if __name__ == '__main__':
    main()
