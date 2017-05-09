from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import itertools
import argparse
import math


class Data:
    def __init__(self, rel, qid, docid, features):
        self.rel = rel
        self.qid = qid
        self.docid = docid
        self.features = features


def sigmoid(x):
    if x < 0:
        return 1.0 - (1.0 / (1.0 + math.exp(x)))
    else:
        return 1.0 / (1.0 + math.exp(-x))


def evaluate(data, data_val, idcg_val, w):
    ndcg_sum = 0
    for qid in data_val:
        dcg = 0
        if idcg_val[qid] != 0:
            for idx, doc in enumerate(sorted([x for rel in data[qid] for x in data[qid][rel]], key=lambda x: sigmoid(np.dot(np.transpose(w), x.features)), reverse=True)[:10]): 
                dcg += (2**doc.rel - 1) / math.log(idx+2, 2) 
            ndcg = dcg / idcg_val[qid] 
            ndcg_sum += ndcg
        else:
            ndcg_sum += 1
    return ndcg_sum / len(data_val)


def L(high, low, data_qid, w):
    tmp = np.zeros(136)
    for x1 in data_qid[high]:
        for x2 in data_qid[low]:
            fx1 = sigmoid(np.dot(np.transpose(w), x1.features))
            fx2 = sigmoid(np.dot(np.transpose(w), x2.features))
            ef = math.exp(fx2-fx1)
            a = ((ef/(1+ef)) * ((fx2*(1-fx2)*x2.features) - (fx1*(1-fx1)*x1.features)))
            tmp += a
    return tmp


def task1(data, eta, data_val, idcg_val, iter):
    #w = np.ones(136) / 136
    np.random.seed(0)
    w = np.random.rand(136)
    w = w / sum(w)
    for it in range(iter):
        print(it, evaluate(data, data_val, idcg_val, w))
        for qid in data:
            tmpL = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(L)(high, low, data[qid], w) for high, low in itertools.combinations(sorted(data[qid].keys(), reverse=True), 2))
            w = w - eta * sum(tmpL)
        tmpL = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(L)(high, low, data, w) for high, low in itertools.combinations(sorted(data['rel'].keys(), reverse=True), 2))
        w = w - eta * sum(tmpL)
        '''
        for high, low in itertools.combinations(sorted(data['rel'].keys(), reverse=True), 2):
            for x1 in data['rel'][high]:
                for x2 in data['rel'][low]:
                    fx1 = sigmoid(np.dot(np.transpose(w), x1.features))
                    fx2 = sigmoid(np.dot(np.transpose(w), x2.features))
                    ef = math.exp(fx2-fx1)
                    w = w - eta * ((ef/(1+ef)) * ((fx2*(1-fx2)*x2.features) - (fx1*(1-fx1)*x1.features)))
        '''
>>>>>>> 5410ff6bf91822780e6d883d974a0e0ec0ab1bcd
    print(evaluate(data, data_val, idcg_val, w))
    print(w)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=int)
    parser.add_argument('-input', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-eta', type=float)
    parser.add_argument('-iter', type=int)
    parser.add_argument('-train', type=str)
    parser.add_argument('-vali', type=str)
    return parser.parse_args()


def dl():
    return defaultdict(list)


def read_data(file_name):
    data = defaultdict(dl)
    with open(file_name) as fp:
        for idx, line in enumerate(fp):
            line = line.strip().split(' ', 3)
            tmp = Data(int(line[0]), int(line[1].split(':')[1]), int(line[2].split(':')[1]), np.array([float(x.split(':')[1]) for x in line[3].split(' ')]))
            data[tmp.qid][tmp.rel].append(tmp)
    return data


def cal_idcg(data):
    idcg = defaultdict(float)
    for qid in data:
        for idx, doc in enumerate(sorted([x for rel in data[qid] for x in data[qid][rel]], key=lambda x: x.rel, reverse=True)[:10]):
            idcg[qid] += ((2**doc.rel) - 1) / math.log(idx+2, 2)
    return idcg


def main():
    args = get_args()
    data = read_data(args.train)
    data_val = read_data(args.vali)
    idcg_val = cal_idcg(data_val)
    if args.task == 1:
        task1(data, args.eta, data_val, idcg_val, args.iter)


if __name__ == '__main__':
    main()
