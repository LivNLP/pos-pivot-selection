import numpy as np
import re
import itertools
from decimal import *
import nltk
import glob
import pickle

# collect words and pos tags from labeled dataset 
def collect_labeled(domain):
    fname = "../data/gweb-%s.conll"%domain
    if domain == "wsj":
        fname = "../data/ontonotes-wsj-train.conll"
    input_file = open(fname,'r')
    words = []
    for line in input_file:
        p = filter(None,re.split("\t",line))
        if p != ["\n"]:
            pos = int(Decimal(p[0]))
            word = p[1].lower()
            tag = p[4]
            words.append([word,tag,pos])
        else:
            words.append(p)
    sentences = split_on_sentences(words)
    # add sentence_length as additional term for generating feature vectors
    new_sentences = [[word+[len(sentence)]  for word in sentence] for sentence in sentences]
    save_name = domain
    if 'wsj' in domain:
        save_name = '%s-labeled'%domain
    save_preprocess_obj(new_sentences,save_name)
    print '%s saved'%save_name 
    pass

# a method to group words into sentences by new lines
def split_on_sentences(lst):
    w=["\n"]
    spl = [list(y) for x, y in itertools.groupby(lst, lambda z: z == w) if not x]
    return spl

# collect data from other unlabeled dataset 
def collect_unlabeled(domain):
    input_file = open("../data/gweb-%s.unlabeled.txt"%domain,"r")
    sentences = []
    for line in input_file:
        p = filter(None,re.split(" ",line.replace("\n","")))
        sentences.append(list(p))
    new_sentences = [[[word.lower()]+['-',sent.index(word)+1,len(sent)] for word in sent] for sent in sentences]
    save_name = '%s-unlabeled'%domain
    save_preprocess_obj(new_sentences,save_name)
    print '%s saved'%save_name, len(new_sentences)
    pass

# collect data from unlabeled wsj
def collect_unlabeled_wsj():
    files = glob.glob('../data/wsj/*/*')
    wsj = []
    for file in files:
        input_file = open(file,'r')       
        next(input_file)
        for line in input_file:
            p = line.rstrip()
            if p: 
                sents = nltk.sent_tokenize(p)
                words = [nltk.word_tokenize(sent) for sent in sents]
                wsj += words
    new_wsj = [[[word.lower()]+['-',sent.index(word)+1,len(sent)] for word in sent] for sent in wsj]
    save_preprocess_obj(new_wsj,'wsj-unlabeled')
    print 'wsj-unlabeled saved'
    pass

def presets_labeled(source,target):

    pass

def feature_list():
    pass

def save_preprocess_obj(obj, name):
    with open('../work/preprocess/'+name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_preprocess_obj(name):
    with open('../work/preprocess/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # domains = ["answers-dev","answers-test","emails-test","reviews-dev"]
    # domains += ["reviews-test","newsgroups-dev","newsgroups-test","weblogs-test"]
    # domains = ["wsj"]
    # for domain in domains:
    #     collect_labeled(domain)
    # # target domain unlabeled datasets
    domains = ["answers","emails","reviews","newsgroups","weblogs"]
    # domain = "answers"
    for domain in domains:
        collect_unlabeled(domain)
    # collect_unlabeled_wsj()