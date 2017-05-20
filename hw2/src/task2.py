from collections import defaultdict
import itertools
import numpy as np
import argparse
import random
import math
import pickle


class Data:
    def __init__(self, rel, qid, docid, features):
        self.rel = rel
        self.qid = qid
        self.docid = docid
        self.features = features


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', type=str)
    parser.add_argument('-test', type=str)
    parser.add_argument('-prefix', type=str)
    parser.add_argument('-lr', type=float)
    parser.add_argument('-reg', type=float)
    parser.add_argument('-T', type=int)
    parser.add_argument('-sample', type=int)
    return parser.parse_args()


def read_and_parse(path):
    data = defaultdict(list)
    fs = list()
    with open(path) as fp:
        for line in fp:
            line = line.strip().split(' ', 3)
            tmpdata = Data(int(line[0]), int(line[1].split(':')[1]), int(line[2].split(':')[1]), np.array([float(x.split(':')[1]) for x in line[3].split(' ')]))
            data[tmpdata.qid].append(tmpdata)
            fs.append(tmpdata.features)
    return data, fs


def scale_data(data, maxx, minx):
    for qid in data:
        for x in data[qid]:
            for i in range(136):
                x.features[i] = (x.features[i] - minx[i]) / (maxx[i] - minx[i])
    return data


def read_data(path_train, path_test):
    data, fs_train = read_and_parse(path_train)
    data_test, fs_test = read_and_parse(path_test)
    fs = np.array(fs_train)
    maxx = np.max(fs, axis=0)
    minx = np.min(fs, axis=0)
    np.save('maxx', maxx)
    np.save('minx', minx)
    data = scale_data(data, maxx, minx)
    data_test = scale_data(data_test, maxx, minx)
    return data, data_test


def split_data(l, n):
    random.shuffle(l)
    pool = list()
    sn = len(l)/n
    for i in range(n):
        pool.append(l[int(sn*i):int(sn*(i+1))])
    for i in range(n):
        yield i, pool[i], list(itertools.chain.from_iterable(pool[:i]+pool[i+1:]))


def f(x, w):
    return np.dot(np.transpose(w), x.features)


def dL(x, w, reg):
    return (-2 * (x.rel - f(x, w)) * x.features + 2 * reg * w) * (x.rel**2 + 1)


def evaluation(data, data_val, w):
    ndcg = 0
    count = 0
    for qid in data_val:
        idcg = 0
        for i, x in enumerate(sorted([x for x in data[qid]], key=lambda x: x.rel, reverse=True)[:10]):
            idcg += ((2 ** x.rel) - 1) / math.log(i+2, 2)
        if idcg != 0:
            dcg = 0
            for i, x in enumerate(sorted([x for x in data[qid]], key=lambda x: f(x, w), reverse=True)[:10]):
                dcg += ((2 ** x.rel) - 1) / math.log(i+2, 2)
            ndcg += dcg / idcg
            count += 1
    return ndcg / count


def mkvali(l):
    random.shuffle(l)
    return l[:500], l[501:]


def train(data, lr, reg, T, sample, prefix):
    w = np.ones(136) / 136
    data_val, data_train = mkvali(list(data.keys()))
    for it in range(T):
        for qid in random.sample(data_train, sample):
            tmpdL = np.zeros(136)
            for x in data[qid]:
                tmpdL += dL(x, w, reg)
            w = w - (lr * tmpdL / len(data[qid]))
        print(it, evaluation(data, data_val, w), '%s_%d' % (prefix, it))
        np.save('result_w/%s_%d' % (prefix, it), w)


def cross_vali_train(data, lr, reg, T, sample, prefix):
    vali_num = 5
    vali = list(split_data(list(data.keys()), vali_num))
    vw = [np.array([1/136]*136)] * vali_num
    w = np.array([1/136]*136)
    for it in range(T):
        ve = 0
        for i, data_val, data_train in vali:
            for qid in data_train:
                tmpdL = np.zeros(136)
                for x in data[qid]:
                    tmpdL += dL(x, vw[i], reg)
                vw[i] = vw[i] - (lr * tmpdL / len(data[qid]))
            ve += evaluation(data, data_val, vw[i])
        for qid in data:
            tmpdL = np.zeros(136)
            for x in data[qid]:
                tmpdL += dL(x, w, reg)
            w = w - (lr * tmpdL / len(data[qid]))
        print(it, ve / vali_num, '%s_%d' % (prefix, it))
        np.save('result_w/%s_%d' % (prefix, it), w)


def main():
    random.seed(1126)
    args = get_args()
    #data, data_test = read_data(args.train, args.test)
    #pickle.dump(data, open('data_train.pickle', 'wb'))
    #pickle.dump(data_test, open('data_test.pickle', 'wb'))
    #pickle.dump((data, data_test), open('data_maxmin.pickle', 'wb'))
    #print('pickle done')
    #data, data_test = pickle.load(open('data_maxmin.pickle', 'rb'))
    data = pickle.load(open('data_train.pickle', 'rb'))
    #train(data, args.lr, args.reg, args.T, args.sample, args.prefix)
    cross_vali_train(data, args.lr, args.reg, args.T, args.sample, args.prefix)


if __name__ == '__main__':
    main()
