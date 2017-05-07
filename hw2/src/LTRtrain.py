from collections import defaultdict
import numpy as np
import itertools
import argparse
import math


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def task1(data, eta):
    w = np.ones(136)

    for _ in range(5):
        for high, low in itertools.combinations(sorted(data.keys(), reverse=True), 2):
            for x1 in data[high]:
                for x2 in data[low]:
                    fx1 = sigmoid(np.dot(np.transpose(w), x1))
                    fx2 = sigmoid(np.dot(np.transpose(w), x2))
                    ef = math.exp(fx2-fx1)
                    w = w - (eta * (ef/1+ef) * ((fx2*(1-fx2)*x2) - (fx1*(1-fx1)*x1)))
        print(w)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=int)
    parser.add_argument('-input', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-eta', type=float)
    return parser.parse_args()


def main():
    args = get_args()
    data = defaultdict(list)

    with open('train.txt') as fp:
    #with open('small.txt') as fp:
        for line in fp:
            line = line.strip().split(' ', 3)
            data[int(line[0])].append(np.array([float(x.split(':')[1]) for x in line[3].split(' ')]))

    if args.task == 1:
        task1(data, args.eta)


if __name__ == '__main__':
    main()
