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
    # add extra empty tag, position, sentence_length to the collected list
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

def presets_unlabeled(source,target):
    # load some preprocessing objects ready
    src_unlabeled = load_preprocess_obj('%s-unlabeled'%source)
    tgt_unlabeled = load_preprocess_obj('%s-unlabeled'%target)

    # number of sentences
    un_src_sentences = len(src_unlabeled)
    un_tgt_sentences = len(tgt_unlabeled)
    un_sentences = un_src_sentences+un_tgt_sentences

    # feature list
    un_src_features = feature_list(src_unlabeled)
    un_tgt_features = feature_list(tgt_unlabeled)
    un_features = set(un_src_features).union(set(un_tgt_features))
    print len(un_features)

    # sentences contain x
    x_un_src = sentences_contain_x(un_features,src_unlabeled)
    print len(x_un_src)
    x_un_tgt = sentences_contain_x(un_features,tgt_unlabeled)
    x_un = combine_dicts(x_un_src, x_un_tgt)
    print len(x_un)

    # save presets to temp objects
    save_obj(source,target,un_src_sentences,"un_src_sentences")
    save_obj(source,target,un_tgt_sentences,"un_tgt_sentences")
    save_obj(source,target,un_sentences,"un_sentences")
    save_obj(source,target,x_un_src,"x_un_src")
    save_obj(source,target,x_un_tgt,"x_un_tgt")
    save_obj(source,target,x_un,"x_un")
    pass

# return a list of all the features in the dataset
def feature_list(sentences):
    return index_list(0,sentences)

# return a list of all the pos tags in the dataset
def tag_list(sentences):
    return index_list(1,sentences)

# general method to return a list of ith value from the given list
def index_list(i,my_list):
    return list(set([a[i] for b in my_list for a in b]))

def sentences_contain_x(features,sentences):
    features = list(features)
    features_bag = np.zeros(len(features), dtype=float)
    # remove the detail we don't need for our computation
    new_sentences = [[word[0] for word in sent] for sent in sentences]
    for sentence in new_sentences:
        for x in set(sentence):
            i = features.index(x)
            features_bag[i] += 1
    return dict(zip(features,features_bag))

# method to combine dictionaries
def combine_dicts(a, b):
    return dict([(n, a.get(n, 0)+b.get(n, 0)) for n in set(a)|set(b)])


# save and load for preprocessing
def save_preprocess_obj(obj, name):
    with open('../work/preprocess/'+name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_preprocess_obj(name):
    with open('../work/preprocess/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)

# save and load by source and target domains
def save_obj(source,target,obj,name):
    with open('../work/%s-%s/'+name + '.pkl'%(source,target), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print '../work/%s-%s/'+name + '.pkl saved'%(source,target)

def load_obj(source,target,name):
    with open('../work/%s-%s/'+name + '.pkl'%(source,target), 'rb') as f:
        return pickle.load(f)


##########test methods##########
def print_test():
    my_object = load_preprocess_obj('answers-dev')
    features = feature_list(my_object)
    print sentences_contain_x(features,my_object)
    pass


if __name__ == "__main__":
    # domains = ["answers-dev","answers-test","emails-test","reviews-dev"]
    # domains += ["reviews-test","newsgroups-dev","newsgroups-test","weblogs-test"]
    # domains = ["wsj"]
    # for domain in domains:
    #     collect_labeled(domain)
    # # target domain unlabeled datasets
    domains = ["answers","emails"]
    domains += ["reviews","newsgroups","weblogs"]
    # domain = "answers"
    # for domain in domains:
    #     collect_unlabeled(domain)
    # collect_unlabeled_wsj()
    source = 'wsj'
    for target in domains:
        presets_unlabeled(source,target)
    # print_test()