import numpy as np
import re
import itertools
from decimal import *
import nltk
import glob
import pickle
import os

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
    # generate a tag list from labeled data for iteration
    src_labeled = load_preprocess_obj('%s-labeled'%source)
    tags = tag_list(src_labeled)
    # tags = ['.']
    # loop tags to divide presets into groups
    for pos_tag in tags:
        print "TAG = %s"% pos_tag
        presets_labeled_tag(source,target,pos_tag,src_labeled)
    pass

# different from SA, for each pos_tag, the source labeled data is divided into
# sentences HAVE pos_tag and NOT pos_tag, so this is a single method for a single pos_tag
# this target does nothing, just for saving into the correct dir
def presets_labeled_tag(source,target,pos_tag,src_labeled):
    # tgt_labeled = load_preprocess_obj('%s-dev'%target)
    src_sentences = float(len(src_labeled))

    # list sentences and number of sentences HAS pos_tag
    pos_src_data = sentence_list_contain_tag(pos_tag,src_labeled)
    pos_src_sentences = float(len(pos_src_data))

    # list sentences and number of sentences NOT pos_tag
    neg_src_data = minus_lists(src_labeled,pos_src_data)
    neg_src_sentences = float(len(neg_src_data))
    print pos_src_sentences,neg_src_sentences

    # feature list
    pos_src_features = feature_list(pos_src_data)
    neg_src_features = feature_list(neg_src_data)
    src_features = set(pos_src_features).union(neg_src_features)
    print len(src_features)

    # sentences contain x
    x_pos_src = sentences_contain_x(src_features,pos_src_data)
    x_neg_src = sentences_contain_x(src_features,neg_src_data)
    x_src = combine_dicts(x_pos_src, x_neg_src)
    print len(x_src)

    # save presets to temp objects
    # pos_tag="TAG."
    save_tag_obj(source,target,pos_src_data,pos_tag,"pos_src_data")
    save_tag_obj(source,target,neg_src_data,pos_tag,"neg_src_data")
    save_tag_obj(source,target,pos_src_sentences,pos_tag,"pos_src_sentences")
    save_tag_obj(source,target,neg_src_sentences,pos_tag,"neg_src_sentences")
    save_tag_obj(source,target,src_features,pos_tag,"src_features")
    save_tag_obj(source,target,x_pos_src,pos_tag,"x_pos_src")
    save_tag_obj(source,target,x_neg_src,pos_tag,"x_neg_src")
    save_tag_obj(source,target,x_src,pos_tag,"x_src")
    pass

def presets_unlabeled(source,target):
    # load some preprocessing objects ready
    src_unlabeled = load_preprocess_obj('%s-unlabeled'%source)
    tgt_unlabeled = load_preprocess_obj('%s-unlabeled'%target)

    # number of sentences
    un_src_sentences = float(len(src_unlabeled))
    un_tgt_sentences = float(len(tgt_unlabeled))
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

# number of sentences contains x
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

# a method to combine dictionaries
def combine_dicts(a, b):
    return dict([(n, a.get(n, 0)+b.get(n, 0)) for n in set(a)|set(b)])

# a method to return a list a-b
def minus_lists(a,b):
    return [x for x in a if x not in b]

# sentences contain pos_tag with position info
def sentence_list_contain_tag(pos_tag,sentences):
    sents_vector = np.zeros(len(sentences), dtype=float)
    for sent in sentences:
        for word in sent:
            if word[1] == pos_tag:
                i = sentences.index(sent)
                sents_vector[i]+=1
    return [sent for sent in sentences if sents_vector[sentences.index(sent)]>0]

# count frequency
def count_freq(sentences,h):
    for sentence in sentences:
        for word in sentence:
            h[word]=h.get(word,0)+1
    pass

# save and load for preprocessing
def save_preprocess_obj(obj, name):
    with open('../work/preprocess/'+name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_preprocess_obj(name):
    with open('../work/preprocess/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)

# save and load by source and target domains
def save_obj(source,target,obj,name):
    filename = '../work/%s-%s/%s.pkl'%(source,target,name)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print '%s saved'%filename

def load_obj(source,target,name):
    with open('../work/%s-%s/%s.pkl'%(source,target,name), 'rb') as f:
        return pickle.load(f)

# save pos_tag related objects in the created subdir
def save_tag_obj(source,target,obj,tag,name):
    filename = '../work/%s-%s/%s/%s.pkl'%(source,target,tag,name)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print filename

def load_tag_obj(source,target,tag,name):
    with open('../work/%s-%s/%s/%s.pkl'%(source,target,tag,name), 'rb') as f:
        return pickle.load(f)


##########test methods##########
def print_test():
    my_object = load_preprocess_obj('wsj-labeled')
    # features = feature_list(my_object)
    print tag_list(my_object),len(tag_list(my_object))
    # print sentence_list_contain_tag('NN', my_object) 
    # print len(my_object), len(sentence_list_contain_tag('NN', my_object))
    pass


if __name__ == "__main__":
    # domains = ["answers-dev","answers-test","emails-test","reviews-dev"]
    # domains += ["reviews-test","newsgroups-dev","newsgroups-test","weblogs-test"]
    # domains = ["wsj"]
    # for domain in domains:
    #     collect_labeled(domain)
    # target domain unlabeled datasets
    # domains = ["answers","emails"]
    # domains += ["reviews","newsgroups","weblogs"]
    # domain = "answers"
    # for domain in domains:
    #     collect_unlabeled(domain)
    # collect_unlabeled_wsj()
    source = 'wsj'
    # for target in domains:
    #     presets_unlabeled(source,target)
    # print_test()
    presets_labeled(source,'answers')