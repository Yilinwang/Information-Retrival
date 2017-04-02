import numpy as np
from argparse import ArgumentParser

def apk(actual, predicted, k=100):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=100):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', type=str)
    args = parser.parse_args()

    actual = {}
    predicted = {}
    with open('queries/ans_train.csv') as fp:
        fp.readline()
        for line in fp:
            tmp = line.strip().split(',')
            actual[tmp[0]] = tmp[1].strip().split(' ')
    with open(args.i) as fp:
        fp.readline()
        for line in fp:
            tmp = line.strip().split(',')
            predicted[tmp[0]] = tmp[1].strip().split(' ')
    if not len(predicted) == len(actual):
        print('not this one')
        exit()
    ans = 0
    for i in actual:
        ans += mapk(actual[i], predicted[i])
    print(ans/10)

if __name__ == '__main__':
    main()
