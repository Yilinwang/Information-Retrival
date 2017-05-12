from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import itertools
import random
import argparse
import math
import pickle


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


def f(w, x):
    return sigmoid(np.dot(np.transpose(w), x.features))


def evaluate(data_val, idcg_val, w):
    ndcg_sum = 0
    count = 0
    for qid in data_val:
        dcg = 0
        if idcg_val[qid] != 0:
            for idx, doc in enumerate(sorted([x for x in data_val[qid]], key=lambda x: f(w, x), reverse=True)[:10]): 
                dcg += (2**doc.rel - 1) / math.log(idx+2, 2)
            ndcg = dcg / idcg_val[qid] 
            ndcg_sum += ndcg
            count += 1
    return ndcg_sum / count


def L(x1, x2, w):
    fx1 = f(w, x1)
    fx2 = f(w, x2)
    ef = math.exp(fx2-fx1)
    return (ef/(1+ef)) * ((fx2*(1-fx2)*x2.features) - (fx1*(1-fx1)*x1.features))


def task1(data, data_rel, eta, data_val, idcg_val, iteration, sample, data_test):
    w = np.random.rand(136)
    for it in range(iteration):
        tmpL = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(L)(high, low, w) for high, low in itertools.product(random.sample(data_rel[1], int(math.sqrt(sample))), random.sample(data_rel[0], int(math.sqrt(sample)))))
        w = w - eta * sum(tmpL)
        print(it, evaluate(data_val, idcg_val, w))

        with open('result_test/task1_eta%.3lf_iter%d_sample%d' % (eta, it+1, sample), 'w') as fp:
            fp.write('QueryId,DocumentId\n')
            for qid in data_test:
                for doc in sorted([x for x in data_test[qid]], key=lambda x: f(w, x), reverse=True)[:10]:
                    fp.write('%d,%d\n' % (doc.qid, doc.docid))

    return w


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=int)
    parser.add_argument('-input', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-eta', type=float)
    parser.add_argument('-iter', type=int)
    parser.add_argument('-train', type=str)
    parser.add_argument('-vali', type=str)
    parser.add_argument('-test', type=str)
    parser.add_argument('-sample', type=int)
    return parser.parse_args()


def read_data(file_name):
    data = defaultdict(list)
    data_rel = defaultdict(list)

    maxx = [-math.inf] * 136
    minx = [math.inf] * 136
    tmpdata = list()

    with open(file_name) as fp:
        for idx, line in enumerate(fp):
            line = line.strip().split(' ', 3)
            tmp = Data(int(line[0]), int(line[1].split(':')[1]), int(line[2].split(':')[1]), np.array([float(x.split(':')[1]) for x in line[3].split(' ')]))

            if len(tmp.features) == 136:
                tmpdata.append(tmp)
                for i in range(136):
                    maxx[i] = max(tmp.features[i], maxx[i])
                    minx[i] = min(tmp.features[i], minx[i])

    print('readdata done')

    for d in tmpdata:
        for i in range(136):
            if minx[i] == 0 and maxx[i] == 0:
                print(n[i])
            else:
                d.features[i] = (d.features[i] + abs(minx[i])) / (maxx[i] + abs(minx[i]))
        data[d.qid].append(d)
        if(d.rel >= 3):
            data_rel[1].append(d)
        else:
            data_rel[0].append(d)

    print('norm done')
                    
    return data, data_rel


def cal_idcg(data):
    idcg = defaultdict(float)
    for qid in data:
        for idx, doc in enumerate(sorted([x for x in data[qid]], key=lambda x: x.rel, reverse=True)[:10]):
            idcg[qid] += ((2**doc.rel) - 1) / math.log(idx+2, 2)
    return idcg


def main():
    args = get_args()

    #data, data_rel = read_data(args.train)
    #data_val = read_data(args.vali)[0]
    #pickle.dump((data, data_rel, data_val), open('data.pickle', 'wb'))
    #data_test = read_data(args.test)[0]
    #pickle.dump(data_test, open('data_test.pickle', 'wb'))
    #print('pickle done')
    data, data_rel, data_val = pickle.load(open('data.pickle', 'rb'))
    data_test = pickle.load(open('data_test.pickle', 'rb'))
    idcg_val = cal_idcg(data_val)

    np.random.seed(0)
    random.seed(0)
    if args.task == 1:
        w = task1(data, data_rel, args.eta, data_val, idcg_val, args.iter, args.sample, data_test)


if __name__ == '__main__':
    main()
