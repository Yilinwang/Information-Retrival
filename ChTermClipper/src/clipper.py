import argparse
import re


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pattern', type=str)
    parser.add_argument('-f', '--filelist', type=str)
    parser.add_argument('-t', '--target', type=str)
    return parser.parse_args()


def read_data(filelist):
    data = [re.split('，|。|；| |\u3000|？|！', line.strip()) for x in open(filelist) for line in open('documents/'+x.strip())]
    data = [x for y in data for x in y if x != '']
    return data


def main():
    args = get_args()
    data = read_data(args.filelist)
    target = Trie([x.strip() for x in open(args.target)])


if __name__ == '__main__':
    main()
