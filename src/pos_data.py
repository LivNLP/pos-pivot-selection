import numpy as np
import re
import itertools
from decimal import *

# collect words and pos tags for experiments
def collect(domain):
    input_file = open("../data/gweb-%s-dev.conll"%domain,'r')
    words = []
    for line in input_file:
        p = filter(None,re.split("\t",line))
        if p != ["\n"]:
            pos = int(Decimal(p[0]))
            word = p[1].lower()
            tag = p[3]
            words.append([pos,word,tag])
        else:
            words.append(p)
    sentences = split_on_sentences(words)
    new_sentences = [word+[len(sentence)] for sentence in sentences for word in sentence]
    print new_sentences
    pass

# a method to group words into sentences by new lines
def split_on_sentences(lst):
    w=["\n"]
    spl = [list(y) for x, y in itertools.groupby(lst, lambda z: z == w) if not x]
    return spl

if __name__ == "__main__":
    domain = "answers"
    collect(domain)