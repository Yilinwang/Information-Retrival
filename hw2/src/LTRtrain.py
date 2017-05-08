from collections import defaultdict
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
        return 1.0 - 1.0 / (1.0 + math.exp(x))
    else:
        return 1.0 / (1.0 + math.exp(-x))


def task1(data, eta, data_val, idcg_val):
    w = np.ones(136) / 136

    for _ in range(5): #iter
        ndcg_sum = 0
        for qid in data_val['rel']:
            dcg = 0
            for idx, doc in enumerate(sorted(data['qid'][qid], key=lambda x: sigmoid(np.dot(np.transpose(w), x.features)), reverse=True)[:10]): 
                dcg += (2**doc.rel - 1) / math.log(idx+2, 2) 
            if idcg_val[qid] != 0:
                ndcg = dcg / idcg_val[qid] 
                ndcg_sum += ndcg
        print(ndcg_sum/10)

        for high, low in itertools.combinations(sorted(data['rel'].keys(), reverse=True), 2):
            for x1 in data['rel'][high]:
                for x2 in data['rel'][low]:
                    fx1 = sigmoid(np.dot(np.transpose(w), x1.features))
                    fx2 = sigmoid(np.dot(np.transpose(w), x2.features))
                    ef = math.exp(fx2-fx1)
                    w = w - (eta * (ef/1+ef) * ((fx2*(1-fx2)*x2.features) - (fx1*(1-fx1)*x1.features)))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=int)
    parser.add_argument('-input', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-eta', type=float)
    return parser.parse_args()


def read_data(file_name):
    data = defaultdict(lambda: defaultdict(list))
    with open(file_name) as fp:
        lines = fp.readlines()
        for idx, line in enumerate(lines):
            line = line.strip().split(' ', 3)
            tmp = Data(int(line[0]), int(line[1].split(':')[1]), int(line[2].split(':')[1]), np.array([float(x.split(':')[1]) for x in line[3].split(' ')]))
            data['rel'][tmp.rel].append(tmp)
            data['qid'][tmp.qid].append(tmp)
    return data


def cal_idcg(data):
    idcg = defaultdict(float)
    for qid in data['qid']:
        for idx, doc in enumerate(sorted(data['qid'][qid], key=lambda x: x.rel, reverse=True)[:10]):
            idcg[qid] += ((2**doc.rel) - 1) / math.log(idx+2, 2)
    return idcg


def main():
    args = get_args()
    data = read_data('small.txt')
    data_val = read_data('small.txt')
    idcg_val = cal_idcg(data_val)
    if args.task == 1:
        task1(data, args.eta, data_val, idcg_val)


if __name__ == '__main__':
    main()
