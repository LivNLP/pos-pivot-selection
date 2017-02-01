#!/usr/bin/env python
# -*- coding: utf-8 -*-

# landmark-based pivot selection method for POS taggimg
# modified from the sentiment classification version for different dataset
# import select_pivots as sp 
import pos_data
import numpy
import pickle
# import gensim,logging
# from glove import Corpus, Glove
# import glove
from cvxopt import matrix
from cvxopt.solvers import qp
import os
import collections
from collections import OrderedDict

# construct training data for such domian: labeled and unlabeled
# format: [[sentence 1],[sentence 2]...]
# not reivews!!! are sentences, so here, we need to deal with sentences
def labeled_sentences(domain_name):
    sentences = pos_data.load_preprocess_obj('%s-labeled'%domain_name)
    return pos_data.format_sentences(sentences)

def labeled_sentences_test(domain_name):
    sentences = pos_data.load_preprocess_obj('%s-test'%domain_name)
    return pos_data.format_sentences(sentences)

def labeled_sentences_dev(domain_name):
    sentences = pos_data.load_preprocess_obj('%s-dev'%domain_name)
    return pos_data.format_sentences(sentences)

def unlabeled_sentences(domain_name):
    sentences = pos_data.load_preprocess_obj('%s-unlabeled'%domain_name)
    return pos_data.format_sentences(sentences)

# word embedding: from word to word vector
# Word2Vec
# trained by two domains: S_L and T_U
def word2vec(source,target):
    sentences = labeled_sentences(source) + unlabeled_sentences(target)
    model = gensim.models.Word2Vec(sentences, min_count=5,workers=4,size=300)
    model.save('../work/%s-%s/word2vec.model' % (source,target))
    return model

# works for both pretrained models
def word_to_vec(feature,model):
    return model[feature]

# GloVe
# trained by two domains: S_L and T_U
def glove(source,target):
    sentences = labeled_sentences(source) + unlabeled_sentences(target)
    corpus_model = Corpus()
    corpus_model.fit(sentences, window=10)
    # corpus_model.save('../work/%s-%s/corpus.model'% (source,target))
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    print('Training the GloVe model')
    model = Glove(no_components=300, learning_rate=0.05)
    model.fit(corpus_model.matrix, epochs=int(10),
              no_threads=6, verbose=True)
    model.add_dictionary(corpus_model.dictionary)
    output_path = '../work/%s-%s/glove.model' % (source,target)
    model.save(output_path)
    # glove_to_word2vec(output_path,output_path+'.gensim')
    return model

# read a pretrained model to get the word vector
def load_pretrained_glove(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model

# filter of pretrained glove model to decrease the memory cost
def load_filtered_glove(source,target,gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    filtered_features = pos_data.load_obj(source,target,'filtered_features')
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        if word in filtered_features:     
            model[word] = embedding
        # if word.replace('.','__') in filtered_features:
        #     model[word.replace('.','__')] = embedding
    print "After filtering, ",len(model)," words loaded!"
    return model

def glove_to_vec(feature,model):
    return model.get_word_vector(feature)

# gamma function: sum up all tags to get one score for single domain pair
def gamma_function(source,target):
    src_labeled = pos_data.load_preprocess_obj('%s-labeled'%source)
    tags = pos_data.tag_list(src_labeled)
    ppmi_dict={}
    for pos_tag in tags:
        ppmi_dict = pos_data.combine_dicts(ppmi_dict,gamma_function_single_tag(source,target,pos_tag))

    dirname = '../work/%s-%s/test/'% (source,target)
    print 'saving ppmi_dict in ' + dirname
    temp = normalize_dict(ppmi_dict)     
    save_loop_obj(temp,dirname,'ppmi_dict')
    pass

# single gamma function by pos_tag: PPMI, other pivot selection methods can be also used
def gamma_function_single_tag(source,target,pos_tag):
    print 'S: %s, T: %s,tag = %s in processing'%(source,target,pos_tag)
    pos_tag = 'TAG.' if pos_tag == '.' else pos_tag
    features = pos_data.load_obj(source,target,'filtered_features')
    x_src = pos_data.load_tag_obj(source,target,pos_tag,"x_src")
    x_pos_src = pos_data.load_tag_obj(source,target,pos_tag,"x_pos_src")
    x_neg_src = pos_data.load_tag_obj(source,target,pos_tag,"x_neg_src")
    pos_src_sentences = pos_data.load_tag_obj(source,target,pos_tag,"pos_src_sentences")
    neg_src_sentences = pos_data.load_tag_obj(source,target,pos_tag,"neg_src_sentences")
    src_sentences = pos_src_sentences+neg_src_sentences

    ppmi_dict = {}
    for x in features:
        if x_src.get(x,0)*x_pos_src.get(x,0)*x_neg_src.get(x,0) > 0:
            pos_pmi = pos_data.pointwise_mutual_info(x_src.get(x,0), x_pos_src.get(x,0), pos_src_sentences, src_sentences) 
            neg_pmi = pos_data.pointwise_mutual_info(x_src.get(x,0), x_neg_src.get(x,0), neg_src_sentences, src_sentences)
            ppmi_dict[x] = (pos_data.ppmi(pos_pmi)-pos_data.ppmi(neg_pmi))**2
        else:
            ppmi_dict[x] = 0
    return ppmi_dict


# f(Wk) = document frequency of Wk in S_L / # documents in S_L -
# document frequency of Wk in T_U / # documents in T_U
def df_diff(df_source,src_sentences,df_target,tgt_sentences):
    return df_source/src_sentences - df_target/tgt_sentences

# uk = f(Wk) * vector(Wk)
# pretrained models with domain specific model support
# Word2Vec
def u_function_pretrained(source,target,model):
    print 'loading objects...'
    # for any tag, x_src is the same.
    df_source = pos_data.load_tag_obj(source,target,'NN','x_src')
    df_target = pos_data.load_obj(source,target,'x_un_tgt')
    src_sentences = float(len(labeled_sentences(source)))
    tgt_sentences = float(len(unlabeled_sentences(target)))
    features = pos_data.load_obj(source,target,'filtered_features')
    ds_model = gensim.models.Word2Vec.load('../work/%s-%s/word2vec.model' % (source,target))

    print 'calculating with pretrained word2vec model...'
    u_dict = {}
    for x in features:
        df_function = df_diff(df_source.get(x,0),src_sentences,df_target.get(x,0),tgt_sentences)
        if x in model.vocab:
            x_vector = word_to_vec(x,model)
        else:
            if x.replace('__','_') in model.vocab:
                print x.replace('__','_')
                x_vector = word_to_vec(x.replace('__','_'),model)
            else:
                x_vector = word_to_vec(x,ds_model)
        u_dict[x] = numpy.dot(df_function,x_vector)

    dirname = '../work/%s-%s/test/'% (source,target)
    print 'saving u_dict_pretrained in ' + dirname
    save_loop_obj(u_dict,dirname,'u_dict_pretrained')
    print 'u_dict_pretrained saved'
    pass

# GloVe
def u_function_pretrained_glove(source,target,model):
    print 'loading objects...'
    # for any tag, x_src is the same.
    df_source = pos_data.load_tag_obj(source,target,'NN','x_src')
    df_target = pos_data.load_obj(source,target,'x_un_tgt')
    src_sentences = float(len(labeled_sentences(source)))
    tgt_sentences = float(len(unlabeled_sentences(target)))
    features = pos_data.load_obj(source,target,'filtered_features')
    dirname = '../work/%s-%s/'% (source,target)
    ds_model = Glove.load(dirname+'glove.model')

    print 'calculating with pretrained glove model...'
    u_dict = {}
    for x in features:
        df_function = df_diff(df_source.get(x,0),src_sentences,df_target.get(x,0),tgt_sentences)
        if model.get(x,0)==0:
            x_vector = glove_to_vec(x,ds_model)
        else:
            x_vector = word_to_vec(x,model)
        u_dict[x] = numpy.dot(df_function,x_vector)

    dirname = '../work/%s-%s/test/'% (source,target)
    print 'saving u_dict_pretrained in ' + dirname
    save_loop_obj(u_dict,dirname,'u_dict_pretrained_glove')
    print 'u_dict_pretrained saved'
    pass


# optimization: QP
def qp_solver(Uk,Rk,param):
    print "u and gamma length: %d, %d" %(len(Uk),len(Rk))
    U = numpy.matrix(sort_by_keys(Uk).values())
    # T = numpy.transpose(U)
    R = sort_by_keys(Rk).values()

    P = numpy.dot(2,U*U.T)
    P = P.astype(float) 
    # print "%d" % len(P)
    # print sort_by_keys(Rk).keys()[:10]
    # print sort_by_keys(Uk).keys()[:10]
    print sort_by_keys(Rk).keys()==sort_by_keys(Uk).keys()
    q = numpy.dot(-param,R)
    n = len(q)
    G = matrix(0.0, (n,n))
    G[::n+1] = -1.0 
    # G = matrix(numpy.identity(n))
    A = matrix(1.0,(1,n))
    h = matrix(0.0,(n,1),tc='d')
    b = matrix(1.0,tc='d')

    solver = qp(matrix(P),matrix(q),G,h,A,b)
    alpha = matrix_to_array(solver['x'])
    s = two_lists_to_dictionary(Uk.keys(),alpha)
    L = s.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    return L

def opt_function(dirname,param,model_name,pretrained):
    print 'loading objects...'
    ppmi_dict = load_loop_obj(dirname,'ppmi_dict')
    if pretrained == 0:
        if model_name == 'word2vec':
            u_dict = load_loop_obj(dirname,'u_dict')
        else:
            u_dict = load_loop_obj(dirname,'u_dict_glove')
    else:
        if model_name == 'word2vec':
            u_dict = load_loop_obj(dirname,'u_dict_pretrained')
        else:
            u_dict = load_loop_obj(dirname,'u_dict_pretrained_glove')

    print 'solving QP...'
    alpha_dict = qp_solver(u_dict,ppmi_dict,param)
    return alpha_dict

# Selecting pivots
# alpha in [0,1], larger is more close to be a landmark (pivot)
def select_pivots_by_alpha(source,target,param,model,pretrained,paramOn):
    temp = 'landmark' if pretrained == 0 else 'landmark_pretrained'
    method = method_name_param(temp,model,param) if paramOn==True else method_name(temp,model,param)
    dirname = '../work/%s-%s/test/'% (source,target)
    L = opt_function(dirname,param,model,pretrained)
    save_loop_obj(L,dirname,method)
    print '%s saved' % method
    # print L[:5]# test
    return L

# helper method
def normalize_dict(a):
    total = sum(a.itervalues(), 0.0)
    a.update((k,v/total) for k,v in a.items())
    return a

def sort_by_keys(d):
    return collections.OrderedDict([(k, d[k]) for k in sorted(d.keys())])

def remove_low_freq_feats(old_dict,new_keys):
    new_dict = {new_key:old_dict[new_key] for new_key in new_keys}
    return new_dict

def freq_keys(source,target,limit):
    src_freq = {}
    tgt_freq = {}
    pos_data.count_freq(labeled_sentences(source), src_freq)
    pos_data.count_freq(unlabeled_sentences(target), tgt_freq) 
    s = {}
    features = set(src_freq.keys()).union(set(tgt_freq.keys()))
    for feat in features:
        temp = min(src_freq.get(feat, 0), tgt_freq.get(feat, 0))
        if temp > limit:
            s[feat] = temp
    L = s.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)   
    return s.keys()

def matrix_to_array(M):
    return numpy.squeeze(numpy.asarray(M))

def two_lists_to_dictionary(keys,values):
    return dict(zip(keys, values))

def method_name(method,word_model,param):
    if param > 10e-3:
        return '%s_%s_ppmi'%(method,word_model)
    else:
        return '%s_%s'%(method,word_model)
    pass

def method_name_param(method,word_model,param):
    return 'test_%s_%s_%f'%(method,word_model,param)

# save and load objects
def load_loop_obj(dirname,name):
    with open(dirname+"%s.pkl" % name,'rb') as f:
        return pickle.load(f)

def save_loop_obj(obj,dirname,name):
    filename = dirname+"%s.pkl" % name
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print filename

def collect_filtered_features(limit):
    source = 'wsj'
    domains = ["answers","emails"]
    domains += ["reviews","newsgroups","weblogs"]
    for target in domains:
        filtered_features = freq_keys(source,target,limit)
        print 'length: %d'% len(filtered_features)
        print 'saving filtered_features for %s-%s ... ' % (source,target)
        pos_data.save_obj(source,target,filtered_features,'filtered_features')
    pass

def create_word2vec_models():
    source = 'wsj'
    domains = ["answers","emails","reviews"]
    domains += ["newsgroups","weblogs"]
    for target in domains:
        print 'creating word2vec model for %s-%s ...' % (source,target) 
        word2vec(source,target)
    print '-----Complete!!-----'
    pass

def create_glove_models():
    source = 'wsj'
    domains = ["answers","emails"]
    domains += ["reviews","newsgroups","weblogs"]
    for target in domains:
        print 'creating GloVe model for %s-%s ...' % (source,target) 
        glove(source,target)
        # print 'calculating u for %s-%s ...' % (source,target)
        # u_function_glove(source,target,model)
    print '-----Complete!!-----'
    pass

def calculate_all_u_pretrained_word2vec():
    # load pretrained model here
    path = '../data/GoogleNews-vectors-negative300.bin'
    model = gensim.models.Word2Vec.load_word2vec_format(path,binary=True)
    # print model.most_similar('very')
    source = 'wsj'
    domains = ["weblogs"]
    # domains += ["answers","emails","reviews","newsgroups"]
    for target in domains:
        print 'calcualting u_pretrained for %s-%s ...' % (source,target)
        u_function_pretrained(source,target,model) 
    print '-----Complete!!-----'
    pass

def calculate_all_u_pretrained_glove():
    # load pretrained model here
    path = '../data/glove.42B.300d.txt'
    # model = load_pretrained_glove(path)
    source = 'wsj'
    domains = ["weblogs"]
    # domains += ["answers","emails","reviews","newsgroups"]
    for target in domains:
        print 'calcualting u_pretrained for %s-%s ...' % (source,target)
        model = load_filtered_glove(source,target,path)
        u_function_pretrained_glove(source,target,model) 
    print '-----Complete!!-----'
    pass

def compute_all_gamma():
    source = 'wsj'
    domains = ["answers","emails"]
    domains += ["reviews","newsgroups","weblogs"]
    for target in domains:
        print 'computing gamma for %s-%s ...' % (source,target)
        gamma_function(source,target)
    print '-----Complete!!-----'
    pass

def store_all_selections(params,model,pretrained,paramOn):
    source = 'wsj'
    domains = ["answers","emails"]
    domains += ["reviews","newsgroups","weblogs"]
    for target in domains:
        for param in params:
            print 'getting alpha from %s-%s ...' % (source,target)
            select_pivots_by_alpha(source,target,param,model,pretrained,paramOn)
            print '------selection completed--------' 
    pass

# test methods
def read_word2vec():
    path = '../data/GoogleNews-vectors-negative300.bin'
    model = gensim.models.Word2Vec.load_word2vec_format(path,binary=True)
    empty_word = numpy.zeros(300, dtype=float)
    print len(numpy.concatenate((model['good'],empty_word)))
    print len(numpy.concatenate((model['good'],model['boy'])))
    pass

# main
if __name__ == "__main__":
    # collect_filtered_features(5)
    # create_word2vec_models()
    # create_glove_models()
    # calculate_all_u_pretrained_word2vec()
    # calculate_all_u_pretrained_glove()
    # compute_all_gamma()
    # params = [0,1]
    # model_names = ['word2vec','glove']
    # ######param#########
    params = [0,0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
    params += [10e-3,10e-4,10e-5,10e-6]
    # model_names = ['word2vec']
    model_names = ['glove']
    paramOn = True
    # paramOn = False
    for model in model_names:
        store_all_selections(params,model,1,paramOn)
    ######test##########
    # read_word2vec()
    # solve_qp() 
    # print_alpha(0)
    # print_ppmi()
