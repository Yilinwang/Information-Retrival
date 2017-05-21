import numpy as np
import pickle
import argparse


class Data:
    def __init__(self, rel, qid, docid, features):
        self.rel = rel
        self.qid = qid
        self.docid = docid
        self.features = features


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=str)
    parser.add_argument('-task', type=int)
    return parser.parse_args()


def f(x, w):
    return np.dot(np.transpose(w), x.features)


def f1(w, x):
    return sigmoid(np.dot(np.transpose(w), x.features))


def main():
    args = get_args()
    #data, data_test = pickle.load(open('data_maxmin.pickle', 'rb'))
    #pickle.dump(data, open('data_train.pickle', 'wb'))
    #pickle.dump(data_test, open('data_test.pickle', 'wb'))
    data_test = pickle.load(open('data_test.pickle', 'rb'))
    w = np.load(args.w)
    if args.task == 1:
        with open('result/%s' % (args.w.strip().split('/')[-1]), 'w') as fp:
            fp.write('QueryId,DocumentId\n')
            for qid in data_test:
                for doc in sorted([x for x in data_test[qid]], key=lambda x: f1(x, w), reverse=True)[:10]:
                    assert doc.qid == qid, 'doc.qid != qid'
                    fp.write('%d,%d\n' % (doc.qid, doc.docid))
    else:
        with open('result/%s' % (args.w.strip().split('/')[-1]), 'w') as fp:
            fp.write('QueryId,DocumentId\n')
            for qid in data_test:
                for doc in sorted([x for x in data_test[qid]], key=lambda x: f(x, w), reverse=True)[:10]:
                    assert doc.qid == qid, 'doc.qid != qid'
                    fp.write('%d,%d\n' % (doc.qid, doc.docid))


if __name__ == '__main__':
    main()
