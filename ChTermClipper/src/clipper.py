from collections import defaultdict
import argparse
import pickle
import string
import re


class Clipper():
    def __init__(self, pre, prefix, suffix, post):
        self.pre = pre
        self.prefix = prefix
        self.suffix = suffix
        self.post = post
        self.T = 0
        self.R = 0

    def __hash__(self):
        return hash((self.pre, self.prefix, self.suffix, self.post))

    def __eq__(self, other):
        return (self.pre == other.pre and self.prefix == other.prefix
                and self.suffix == other.suffix and self.post == other.post)

    def __str__(self):
        return '%s,%s,%s,%s,T=%d,R=%d,score=%lf' % (self.pre, self.prefix, self.suffix, self.post, self.T, self.R, (self.R*self.R)/self.T)


class Trie():
    def __init__(self, words):
        self.root = dict()
        for w in words:
            cur_dict = self.root
            for ch in w:
                cur_dict = cur_dict.setdefault(ch, {})
            cur_dict['_end_'] = '_end_'
    
    def __contains__(self, word):
        cur_dict = self.root
        for ch in word:
            if ch in cur_dict:
                cur_dict = cur_dict[ch]
            else:
                return False
        if '_end_' in cur_dict:
            return True
        return False

    def find(self, sentance):
        cur_dict = self.root
        cur_word = ''
        result = []
        for ch in sentance:
            if ch in cur_dict:
                if '_end_' in cur_dict:
                    result.append(cur_word)
                cur_word += ch
                cur_dict = cur_dict[ch]
            else:
                break
        if '_end_' in cur_dict:
            result.append(cur_word)
        return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pattern', type=str)
    parser.add_argument('-f', '--filelist', type=str)
    parser.add_argument('-t', '--target', type=str)
    return parser.parse_args()


def parse_pattern(raw_pattern):
    p = raw_pattern.strip().split(',')
    assert len(p) == 4, 'pattern format: x,x,x,x'
    return (int(p[0]), int(p[1]), int(p[2]), int(p[3]))


def replace_punctuation(x):
    punctuation = string.punctuation + '。？！，；：「」『』（）…‧《》〈〉'
    for c in punctuation:
        x = x.replace(c, '⊥')
    return x


def read_data(filelist, pattern):
    data = [re.split('。|；| |\u3000|？|！', line.strip()) for x in open(filelist) for line in open('documents/'+x.strip())]
    padding = '@'*max(pattern[0], pattern[3])
    data = [padding+replace_punctuation(x)+padding for y in data for x in y if x != '']
    return data


def score_clipper(data, clipper, pattern, target):
    clipper_index = defaultdict(list)
    for key in clipper:
        for i, c in enumerate(clipper[key]):
            clipper_index[c.pre[0]].append((key, i))

    for sentance in data:
        for ich, ch in enumerate(sentance):
            for key, i in clipper_index[ch]:
                c = clipper[key][i]
                p = (r'%s%s.*%s%s' % (c.pre, c.prefix, c.suffix, c.post))
                for left, right in set([(x.start(), x.end()) for x in re.finditer(p, sentance[ich:])]):
                    clipper[key][i].T += 1
                    if sentance[ich+left+pattern[0]:ich+right-pattern[3]] in target:
                        clipper[key][i].R += 1
    return clipper


def find_clipper(data, target, pattern):
    clipper = defaultdict(list)
    for sentance in data:
        for index in range(len(sentance)):
            result = target.find(sentance[index:])
            for x in result:
                clipper[x].append(Clipper(sentance[index-pattern[0]:index], sentance[index: index+pattern[1]], sentance[index+len(x)-pattern[2]:index+len(x)], sentance[index+len(x):index+len(x)+pattern[3]]))
    for key in clipper:
        clipper[key] = list(set(clipper[key]))
    return score_clipper(data, clipper, pattern, target)


def main():
    args = get_args()
    pattern = parse_pattern(args.pattern)
    data = read_data(args.filelist, pattern)
    target = Trie([x.strip() for x in open(args.target)])
    
    clipper = find_clipper(data, target, pattern)
    for key in clipper:
        for c in clipper[key]:
            print(key, c)
    pickle.dump(clipper, open('clipper.pickle', 'wb'))


if __name__ == '__main__':
    main()
