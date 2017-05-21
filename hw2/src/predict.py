from collections import defaultdict
import numpy as np
import pickle
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
        

def read_data1(file_name):
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

    for d in tmpdata:
        for i in range(136):
            if minx[i] == 0 and maxx[i] == 0:
                print(n[i])
            else:
                d.features[i] = (d.features[i] - minx[i]) / (maxx[i] - minx[i])
        data[d.qid].append(d)
        if(d.rel >= 3):
            data_rel[1].append(d)
        else:
            data_rel[0].append(d)
    return data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=str)
    parser.add_argument('-input', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-task', type=int)
    return parser.parse_args()


def f(x, w):
    return np.dot(np.transpose(w), x.features)


def f1(x, w):
    return sigmoid(np.dot(np.transpose(w), x.features))


def read_and_parse(path):
    data = defaultdict(list)
    fs = list()
    with open(path) as fp:
        for line in fp:
            line = line.strip().split(' ', 3)
            tmpdata = Data(int(line[0]), int(line[1].split(':')[1]), int(line[2].split(':')[1]), np.array([float(x.split(':')[1]) for x in line[3].split(' ')]))
            data[tmpdata.qid].append(tmpdata)
            fs.append(tmpdata.features)
    return data


def scale_data(data, maxx, minx):
    for qid in data:
        for x in data[qid]:
            for i in range(136):
                x.features[i] = (x.features[i] - minx[i]) / (maxx[i] - minx[i])
    return data


def main():
    args = get_args()
    #data, data_test = pickle.load(open('data_maxmin.pickle', 'rb'))
    #pickle.dump(data, open('data_train.pickle', 'wb'))
    #pickle.dump(data_test, open('data_test.pickle', 'wb'))
    #data_test = pickle.load(open('data_test.pickle', 'rb'))
    if args.task == 1:
        w = np.load('model/task1_w')
        data = read_data1(args.input)
        with open(args.output, 'w') as fp:
            fp.write('QueryId,DocumentId\n')
            for qid in data:
                for doc in sorted([x for x in data[qid]], key=lambda x: f1(x, w), reverse=True)[:10]:
                    assert doc.qid == qid, 'doc.qid != qid'
                    fp.write('%d,%d\n' % (doc.qid, doc.docid))

    else:
        if args.task == 2:
            w = np.load('model/task2_w')
        elif args.task == 3:
            w = np.load('model/task3_w')
        maxx = np.load('model/maxx')
        minx = np.load('model/minx')
        data = read_and_parse(args.input)
        data = scale_data(data, maxx, minx)
        with open(args.output, 'w') as fp:
            fp.write('QueryId,DocumentId\n')
            for qid in data:
                for doc in sorted([x for x in data[qid]], key=lambda x: f(x, w), reverse=True)[:10]:
                    assert doc.qid == qid, 'doc.qid != qid'
                    fp.write('%d,%d\n' % (doc.qid, doc.docid))


if __name__ == '__main__':
    main()
