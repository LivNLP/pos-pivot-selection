'''
Forked Code from Danushka Bollegala
Implementation of SCL following steps after pivot selection
Used for evaluation of pivot selection methods
-----------

Change log: made some changes to do a multilabel classification
for Cross-domain POS tagging, and evaluation on selection of training features
'''

import numpy as np
import scipy.io as sio 
import scipy.sparse as sp
from sparsesvd import sparsesvd

import sys, math, subprocess, time

import pos_data
import classify_pos
import re
import scipy.stats

# import sklearn
import os

def clopper_pearson(k,n,alpha=0.05):
    '''
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    '''
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi

def trainLBFGS(train_file, model_file):
    '''
    Train lbfgs on train file. and evaluate on test file.
    Read the output file and return the classification accuracy.
    '''
    retcode = subprocess.call(
        'classias-train -tb -a lbfgs.logistic -pc1=0 -pc2=1 -m %s %s > /dev/null'  %\
        (model_file, train_file), shell=True)
    # retcode = subprocess.call('~/liblinear-multicore-2.11-1/train -s 0 -n 8 %s %s' %\
    #     (train_file,model_file), shell=True)
    return retcode


def trainMultiLBFGS(train_file, model_file):
    
    # retcode = subprocess.call(
    #     'classias-train -tn -a truncated_gradient.logistic -m %s %s > /dev/null'  %\
    #     (model_file, train_file), shell=True)

    retcode = subprocess.call('~/liblinear-multicore-2.11-1/train -s 0 -n 8 %s %s > /dev/null' %\
        (train_file,model_file), shell=True)
    # LR = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1)
    # model_file= LR.fit(train_file,None)
    return retcode


def testLBFGS(test_file, model_file):
    output = '../work/output'
    # retcode = subprocess.call('cat %s | classias-tag -m %s -t> %s' %\
    #                           (test_file, model_file, output), shell=True)
    # retcode = subprocess.check_output(['/bin/bash','-i', '-c', 'liblinear-predict -b 1 %s %s %s' %\
    #     (test_file,model_file,output)])
    retcode = subprocess.check_output('~/liblinear-multicore-2.11-1/predict %s %s %s' %\
        (test_file,model_file,output), shell=True)
    # retcode = subprocess.check_output('~/liblinear-new/predict %s %s %s' %\
    #     (test_file,model_file,output), shell=True)
    line = retcode
    accuracy = 0
    correct = 0
    total = 0
    p = line.strip().split()
    accuracy = float(p[2].strip('%'))/100
    [correct, total]=[int(s) for s in re.findall(r'\b\d+\b',p[3])]
    # print accuracy,correct,total
    return accuracy,correct,total


def loadClassificationModel(modelFileName):
    '''
    Read the model file and return a list of (feature, weight) tuples.
    '''
    modelFile = open(modelFileName, 'r') 
    weights = []
    for line in modelFile:
        if line.startswith('@'):
            # this is @classias or @bias. skip those.
            continue
        p = line.strip().split()
        featName = p[1].strip()
        featVal = float(p[0])
        if featName == '__BIAS__':
            # This is the bias term
            bias = featVal
        else:
            # This is an original feature.
            if featVal > 0:
                weights.append((featName, featVal))
    modelFile.close()
    return weights

def selectTh(h, t):
    '''
    Select all elements of the dictionary h with frequency greater than t. 
    '''
    p = {}
    for (key, val) in h.iteritems():
        if val > t:
            p[key] = val
    del(h)
    return p


def learnProjection(sourceDomain, targetDomain, pivotsMethod, n):
    '''
    Learn the projection matrix and store it to a file. 
    '''
    h = 50 # no. of SVD dimensions.
    nEmbed = 1500
    #n = 500 # no. of pivots.

    # Parameters to reduce the number of features in the tail
    domainTh = {'wsj':5, 'answers':5, 'emails':5, 'reviews':5, 'weblogs':5,'newsgroups':5}

    # Load pivots.
    features = pos_data.load_obj(sourceDomain,targetDomain,pivotsMethod) if 'landmark' not in pivotsMethod else pos_data.load_obj(sourceDomain,targetDomain,'/test/'+pivotsMethod)
    pivots = dict(features[:n]).keys()
    print 'selecting top-%d features in %s as pivots' % (n, pivotsMethod)

    # Load features and get domain specific features
    features = pos_data.load_obj(sourceDomain,targetDomain,'un_freq') if 'un_' in pivotsMethod else pos_data.load_obj(sourceDomain,targetDomain,'freq')
    print len(features)
    feats = selectTh(dict(features),domainTh[sourceDomain])
    feats = feats.keys()
    print len(feats)
    if 'landmark' in pivotsMethod:
        feats = pos_data.load_obj(sourceDomain,targetDomain,'filtered_features')
    # print 'experimental features = ', len(feats)

    # DSwords = [item for item in feats if item not in pivots]

    
    # Load train vectors.
    print 'Loading Training vectors...',
    startTime = time.time()
    vects = []
    vects.extend(loadFeatureVecors(pos_data.load_preprocess_obj('%s-labeled'%sourceDomain), feats))
    vects.extend(loadFeatureVecors(pos_data.load_preprocess_obj('%s-unlabeled'%sourceDomain), feats))
    vects.extend(loadFeatureVecors(pos_data.load_preprocess_obj('%s-unlabeled'%targetDomain), feats))
    endTime = time.time()
    print '%ss' % str(round(endTime-startTime, 2))     
    # print len([word for v in vects for word in v])

    print 'Total no. of documents =', len(vects)
    print 'Total no. of features =', len(feats)

    # Learn pivot predictors.
    count = 0
    print 'Learning Pivot Predictors..'
    startTime = time.time()
    # M = sp.lil_matrix((len(feats), len(pivots)), dtype=np.float)
    M = np.zeros(shape=(len(feats), len(pivots)))
    for (j, w) in enumerate(pivots):
        # print '%d of %d %s' % (j, len(pivots), w)
        for (feat, val) in getWeightVector(w, vects):
            i = feats.index(feat)
            # i = feat
            M[i,j] = val
        if np.sum(M[:,j])==0:
            count+=1;
            print "*******zero column!******"
    # check null rows in M
    null_rows = sum([1 for row in M if np.sum(row)==0])
    print M.shape, count,null_rows
    # f = open("temp","w")
    # f.write(M)
    # f.close()
    endTime = time.time()
    print 'Took %ss' % str(round(endTime-startTime, 2)) 
    # save_M = '../work/%s/%s-%s/%s/proj.mat' % (sourceDomain, targetDomain,pivotsMethod)  
    # classify_pos.save_loop_obj(M,save_M,'weight_matrix')

    # print 'Loading Word Embeddings..'
    # M2 = sp.lil_matrix(len(feats),len(nEmbed))
    # M = np.concatenate((M,M2))

    performSVD(M, pivotsMethod,sourceDomain, targetDomain,h)
    pass

# separate the step of perform SVD
def performSVD(M,method,sourceDomain, targetDomain,h):
    # Perform SVD on M, M can be weight matrix or can be a combination of weight matrix & embeddings  
    print 'Perform SVD on the matrix...'
    startTime = time.time()
    # ut, s, vt = sparsesvd(M.tocsc(), h)
    ut, s, vt = np.linalg.svd(M)
    print ut.shape
    ut = ut[:h]
    print ut.shape
    endTime = time.time()
    print '%ss' % str(round(endTime-startTime, 2))   
    filename = '../work/%s/%s-%s/proj.mat' % (method,sourceDomain, targetDomain)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    sio.savemat(filename, {'proj':ut.T})
    pass


def getWeightVector(word, vects):
    '''
    Train a binary classifier to predict the given word and 
    return the corresponding weight vector. Not depending on the task,
    always a binary classification, a pivot or not a pivot.
    '''
    trainFileName = '../work/trainFile'
    modelFileName = '../work/modelFile'
    trainFile = open(trainFileName, 'w')
    # feats=list(set([word for v in vects for word in v]))
    
    # for v in vects:
    #     fv = v.copy()
    #     iv = [feats.index(w) for w in fv]
    #     # print iv 
    #     iw = feats.index(word)
    #     # print iw
    #     if word in fv:
    #         label = 1
    #         fv.remove(word)
    #         iv.remove(iw)
    #     else:
    #         label = -1
    #     trainFile.write('%d %s\n'% label,' '.join(str('%d:1'%idx) for idx in iv))
    #     # for idx in iv:
    #     #     trainFile.write('%d:1 '%idx)
    #     # trainFile.write('\n')
    # trainFile.close()
    # trainLBFGS(trainFileName, modelFileName)
    # return loadClassificationModel(modelFileName)
    for v in vects:
        fv = v.copy()
        if word in fv:
            label = 1
            fv.remove(word)
        else:
            label = -1
        trainFile.write('%d %s\n' % (label, ' '.join(fv)))
    trainFile.close()
    trainLBFGS(trainFileName, modelFileName)
    return loadClassificationModel(modelFileName)


def loadFeatureVecors(sentences, feats):
    '''
    Returns a list of lists that contain features for a document. 
    '''
    # sentences = pos_data.format_sentences(old_sentences)
    L = []
    for sent in sentences:
        L.append(set([word[0] for word in sent])&(set(feats)))
    # print L
    return L

# a function for split list to list of lists
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]


def evaluate_POS(source, target, project, gamma, method, n):
    '''
    Report the cross-domain POS classification accuracy. 
    '''
    # Parameters to reduce the number of features in the tail
    print 'lexical features+word embeddings for %s-%s' % (source,target)
    domainTh = {'wsj':5, 'answers':5, 'emails':5, 'reviews':5, 'weblogs':5,'newsgroups':5}

    # gamma = 1.0
    nEmbed = 1500
    window_size = 5
    print 'Source Domain', source
    print 'Target Domain', target
    if project:
        print 'Projection ON', 'Gamma = %f' % gamma
    else:
        print 'Projection OFF'
    # Load the projection matrix.
    M = sp.csr_matrix(sio.loadmat('../work/%s/%s-%s/proj.mat' % (method,source, target))['proj'])
    (nDS, h) = M.shape
    print M.shape

    # Load pivots.
    features = pos_data.load_obj(source,target,method) if 'landmark' not in method else pos_data.load_obj(source,target,'/test/'+method)
    pivots = dict(features[:n]).keys()
    print 'selecting top-%d features in %s as pivots' % (n, method)

    # Load features
    features = pos_data.load_obj(source,target,'un_freq') if 'un_' in method else pos_data.load_obj(source,target,'freq')
    feats = selectTh(dict(features),domainTh[source])
    feats = feats.keys()
    if 'landmark' in method:
        feats = pos_data.load_obj(source,target,'filtered_features')
    print 'experimental features = ', len(feats)
    # DSwords = [item for item in feats if item not in pivots]

    
    # write train feature vectors.
    trainFileName = '../work/%s/%s-%s/trainVects.SCL' % (method, source, target)
    testFileName = '../work/%s/%s-%s/testVects.SCL' % (method, source, target)
    # featFile = open(trainFileName, 'w')
    
    train_sentences = pos_data.load_preprocess_obj('%s-labeled'%source)
    train_vectors = classify_pos.load_classify_obj('%s-labeled-classify'%source)
    # load lexical features as additional features
    train_feats = classify_pos.load_classify_obj('%s-labeled-lexical'%source)
    test_sentences = pos_data.load_preprocess_obj('%s-test'%target)
    test_vectors = classify_pos.load_classify_obj('%s-test-classify'%target)
    # load lexical features as additional features
    test_feats = classify_pos.load_classify_obj('%s-test-lexical'%target)
    tag_list = list(set(pos_data.tag_list(train_sentences))&set(pos_data.tag_list(test_sentences)))
    print 'number of tags = ',len(tag_list)

    # for nSent,sent in enumerate(train_sentences):
    #     words = [word[0] for word in sent]
    #     for nWord,w in enumerate(words):
    #         pos_tag = sent[nWord][1]
    #         if pos_tag in tag_list:
    #             featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
    #             x = sp.lil_matrix((1, nDS), dtype=np.float64)
    #             lex = train_feats[nSent][nWord]
    #             for ft in lex:
    #                 if ft[0]!=0 and ft[0] in feats:
    #                     x[0,feats.index(ft[0])] =ft[1] 
    #             if project:
    #                 y = x.tocsr().dot(M)
    #                 for i in range(0, h):
    #                     featFile.write('%d:%f ' % (i+1, gamma * y[0,i])) 
    #             z = train_vectors[nSent][nWord]
    #             word_vectors=split_list(z,window_size)
    #             for word_index,word_vec in enumerate(word_vectors):
    #                 for i,num in enumerate(word_vec):
    #                     if num != 0:
    #                         featFile.write('%d:%f ' % (((word_index+1)*1000+i),num)) 
    #                 featFile.write('%s'%(' '.join(str('%d:%d'%(((word_index+1)*1000+i),num)) for i,num in enumerate(word_vec) if num != 0)))
                        
                # lex = train_feats[nSent][nWord]
                # for ft in lex:
                #     featFile.write('%s:%f ' % (ft[0],ft[1])) 
                # print 'word %d of %d, sentence %d of %d...'%(nWord,len(words),nSent,len(train_sentences))
    #             featFile.write('\n')
    # featFile.close()
    featFile = open(testFileName, 'w')
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = sp.lil_matrix((1, nDS), dtype=np.float64)
                lex = test_feats[nSent][nWord]
                for ft in lex:
                    if ft[0]!=0 and ft[0] in feats:
                        x[0,feats.index(ft[0])] =ft[1] 
                if project:
                    y = x.tocsr().dot(M)
                    for i in range(0, h):
                        featFile.write('%d:%f ' % (i+1, gamma * y[0,i])) 
                z = test_vectors[nSent][nWord]
                word_vectors=split_list(z,window_size)
                for word_index,word_vec in enumerate(word_vectors):
                    for i,num in enumerate(word_vec):
                        if num != 0:
                            featFile.write('%d:%f ' % (((word_index+1)*1000+i),num)) 
                featFile.write('\n')
    featFile.close()
    # # Train using classias.
    # print 'Training...'
    # modelFileName = '../work/%s/%s-%s/model.SCL.%f' % (method, source, target,gamma)
    # trainMultiLBFGS(trainFileName, modelFileName)
    # # Test using classias.
    # print 'Testing...'
    # [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    # intervals = clopper_pearson(correct,total)
    # print 'Accuracy =', acc
    # print 'Intervals=', intervals
    # print '###########################################\n\n'
    # return acc,intervals

def evaluate_POS_lexical(source, target, project, gamma, method, n):
    '''
    Report the cross-domain POS classification accuracy. 
    not using word embeddings
    '''
    # Parameters to reduce the number of features in the tail
    print 'lexical features for %s-%s' % (source,target)
    domainTh = {'wsj':5, 'answers':5, 'emails':5, 'reviews':5, 'weblogs':5,'newsgroups':5}

    # gamma = 1.0
    nEmbed = 1500
    window_size = 5
    print 'Source Domain', source
    print 'Target Domain', target
    if project:
        print 'Projection ON', 'Gamma = %f' % gamma
    else:
        print 'Projection OFF'
    # Load the projection matrix.
    M = sp.csr_matrix(sio.loadmat('../work/%s/%s-%s/proj.mat' % (method,source, target))['proj'])
    (nDS, h) = M.shape

    # Load pivots.
    features = pos_data.load_obj(source,target,method) if 'landmark' not in method else pos_data.load_obj(source,target,'/test/'+method)
    pivots = dict(features[:n]).keys()
    print 'selecting top-%d features in %s as pivots' % (n, method)

    # Load features
    features = pos_data.load_obj(source,target,'un_freq') if 'un_' in method else pos_data.load_obj(source,target,'freq')
    feats = selectTh(dict(features),domainTh[source])
    feats = feats.keys()
    if 'landmark' in method:
        feats = pos_data.load_obj(source,target,'filtered_features')
    print 'experimental features = ', len(feats)
    # DSwords = [item for item in feats if item not in pivots]

    
    # write train feature vectors.
    trainFileName = '../work/%s/%s-%s/trainVects_lexical.SCL' % (method,source, target)
    testFileName = '../work/%s/%s-%s/testVects_lexical.SCL' % (method,source, target)
    featFile = open(trainFileName, 'w')
    
    train_sentences = pos_data.load_preprocess_obj('%s-labeled'%source)
    # train_vectors = classify_pos.load_classify_obj('%s-labeled-classify'%source)
    # load lexical features as additional features
    train_feats = classify_pos.load_classify_obj('%s-labeled-lexical'%source)
    test_sentences = pos_data.load_preprocess_obj('%s-test'%target)
    # test_vectors = classify_pos.load_classify_obj('%s-test-classify'%target)
    # load lexical features as additional features
    test_feats = classify_pos.load_classify_obj('%s-test-lexical'%target)
    tag_list = list(set(pos_data.tag_list(train_sentences))&set(pos_data.tag_list(test_sentences)))
    print 'number of tags = ',len(tag_list)

    for nSent,sent in enumerate(train_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = sp.lil_matrix((1, nDS), dtype=np.float64)
                lex = train_feats[nSent][nWord]
                for ft in lex:
                    if ft[0]!=0 and ft[0] in feats:
                        x[0,feats.index(ft[0])] =ft[1] 
                if project:
                    y = x.tocsr().dot(M)
                    for i in range(0, h):
                        featFile.write('%d:%f ' % (i+1, gamma * y[0,i])) 
                featFile.write('\n')
    featFile.close()
    featFile = open(testFileName, 'w')
    
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = sp.lil_matrix((1, nDS), dtype=np.float64)
                lex = test_feats[nSent][nWord]
                for ft in lex:
                    if ft[0]!=0 and ft[0] in feats:
                        x[0,feats.index(ft[0])] =ft[1]
                if project:
                    y = x.tocsr().dot(M)
                    for i in range(0, h):
                        featFile.write('%d:%f ' % (i+1, gamma * y[0,i])) 
                featFile.write('\n')
    featFile.close()
    # Train using classias.
    print 'Training...'
    modelFileName = '../work/%s/%s-%s/model_lexical.SCL' % (method,source, target)
    trainMultiLBFGS(trainFileName, modelFileName)
    # Test using classias.
    print 'Testing...'
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print 'Accuracy =', acc
    print 'Intervals=', intervals
    print '###########################################\n\n'
    return acc,intervals



def evaluate_POS_NA(source,target):
    print 'word embeddings for %s-%s noAdapt' % (source,target)
    # trainFileName = '../work/%s-%s/trainVects.NA' % (source, target)
    testFileName = '../work/%s-%s/testVects.NA' % (source, target)
    # featFile = open(trainFileName, 'w')
    window_size = 5

    # train_sentences = pos_data.load_preprocess_obj('%s-labeled'%source)
    # train_vectors = classify_pos.load_classify_obj('%s-labeled-classify'%source)
    # print 'training features = ', len([word for sent in train_sentences for word in sent])
    # test_sentences = pos_data.load_preprocess_obj('%s-test'%target)
    # test_vectors = classify_pos.load_classify_obj('%s-test-classify'%target)
    # print 'test features =', len([word for sent in test_sentences for word in sent])
    # tag_list = list(set(pos_data.tag_list(train_sentences))&set(pos_data.tag_list(test_sentences)))
    # print 'number of tags = ',len(tag_list)
    # for nSent,sent in enumerate(train_sentences):
    #     words = [word[0] for word in sent]
    #     for nWord,w in enumerate(words):
    #         pos_tag = sent[nWord][1]
    #         if pos_tag in tag_list:
    #             featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
    #             x = train_vectors[nSent][nWord]
    #             word_vectors=split_list(x,window_size)
    #             for word_index,word_vec in enumerate(word_vectors):
    #                 for i,num in enumerate(word_vec):
    #                     if num != 0:
    #                         featFile.write('%d:%f ' % (((word_index+1)*1000+i),num)) 
    #             featFile.write('\n')
    # featFile.close()
    featFile = open(testFileName, 'w')
    
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = test_vectors[nSent][nWord]
                word_vectors=split_list(x,window_size)
                for word_index,word_vec in enumerate(word_vectors):
                    for i,num in enumerate(word_vec):
                        if num != 0:
                            featFile.write('%d:%f ' % (((word_index+1)*1000+i),num)) 
                featFile.write('\n')
    featFile.close()
    # Train using classias.
    # modelFileName = '../work/%s-%s/model.NA' % (source, target)
    # print 'Training...'
    # trainMultiLBFGS(trainFileName, modelFileName)
    # # Test using classias.
    # print 'Testing...'
    # [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    # intervals = clopper_pearson(correct,total)
    # print 'Accuracy =', acc
    # print 'Intervals=', intervals
    # print '###########################################\n\n'
    # return acc,intervals
    pass


def evaluate_POS_NA_lexical(source,target):
    print 'lexical features for %s-%s noAdapt' % (source,target)
    trainFileName = '../work/%s-%s/trainVects_lexical.NA' % (source, target)
    testFileName = '../work/%s-%s/testVects_lexical.NA' % (source, target)
    featFile = open(trainFileName, 'w')
    window_size = 5
    train_sentences = pos_data.load_preprocess_obj('%s-labeled'%source)
    train_feats = classify_pos.load_classify_obj('%s-labeled-lexical'%source)
    print 'training features = ', len([word for sent in train_sentences for word in sent])
    test_sentences = pos_data.load_preprocess_obj('%s-test'%target)
    test_feats = classify_pos.load_classify_obj('%s-test-lexical'%target)
    print 'test features = ', len([word for sent in test_sentences for word in sent])
    feat_list = set([word for sent in train_sentences for word in sent])&set([word for sent in test_sentences for word in sent])
    tag_list = list(set(pos_data.tag_list(train_sentences))&set(pos_data.tag_list(test_sentences)))
    print 'number of tags = ',len(tag_list)
    for nSent,sent in enumerate(train_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = train_feats[nSent][nWord]
                for ft in x:
                    if ft[1] != 0:
                        featFile.write('%d:%f ' % (feat_list.index(ft[0]),ft[1])) 
                featFile.write('\n')
    featFile.close()
    featFile = open(testFileName, 'w')
   
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = test_feats[nSent][nWord]
                for ft in x:
                    if ft[1] != 0:
                        featFile.write('%d:%f ' % (feat_list.index(ft[0]),ft[1])) 
                featFile.write('\n')
    featFile.close()
    # Train using classias.
    modelFileName = '../work/%s-%s/model_lexical.NA' % (source, target)
    print 'Training...'
    trainMultiLBFGS(trainFileName, modelFileName)
    # Test using classias.
    print 'Testing...'
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print 'Accuracy =', acc
    print 'Intervals=', intervals
    print '###########################################\n\n'
    return acc,intervals
    pass



def evaluate_POS_ID(source):
    print 'word embeddings for %s inDomain' % (source)
    trainFileName = '../work/%s/trainVects.ID' % (source)
    testFileName = '../work/%s/testVects.ID' % (source)
    featFile = open(trainFileName, 'w')
    window_size = 5 
    train_sentences = pos_data.load_preprocess_obj('%s-dev'%source)
    train_vectors = classify_pos.load_classify_obj('%s-dev-classify'%source)
    print 'training features = ', len([word for sent in train_sentences for word in sent])
    test_sentences = pos_data.load_preprocess_obj('%s-test'%source)
    test_vectors = classify_pos.load_classify_obj('%s-test-classify'%source)
    print 'test features = ', len([word for sent in test_sentences for word in sent])
    tag_list = list(set(pos_data.tag_list(train_sentences))&set(pos_data.tag_list(test_sentences)))
    print 'number of tags = ',len(tag_list)
    print 'Loading training vectors...'
    for nSent,sent in enumerate(train_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = train_vectors[nSent][nWord]
                word_vectors=split_list(x,window_size)
                for word_index,word_vec in enumerate(word_vectors):
                    for i,num in enumerate(word_vec):
                        if num != 0:
                            featFile.write('%d:%f ' % (((word_index+1)*1000+i),num))  
                featFile.write('\n')
    featFile.close()
    featFile = open(testFileName, 'w')
    print 'Loading test vectors...'
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = test_vectors[nSent][nWord]
                word_vectors=split_list(x,window_size)
                for word_index,word_vec in enumerate(word_vectors):
                    for i,num in enumerate(word_vec):
                        if num != 0:
                            featFile.write('%d:%f ' % (((word_index+1)*1000+i),num)) 
                featFile.write('\n')
    featFile.close()
    # Train using classias.
    modelFileName = '../work/%s/model.ID' % (source)
    trainMultiLBFGS(trainFileName, modelFileName)
    print 'training finished'
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print 'Accuracy =', acc
    print 'Intervals=', intervals
    print '###########################################\n\n'
    return acc,intervals
    pass


def evaluate_POS_ID_lexical(source):
    print 'lexical features for %s inDomain' % (source)
    trainFileName = '../work/%s/trainVects_lexical.ID' % (source)
    testFileName = '../work/%s/testVects_lexical.ID' % (source)
    featFile = open(trainFileName, 'w')
    train_sentences = pos_data.load_preprocess_obj('%s-dev'%source)
    train_feats = classify_pos.load_classify_obj('%s-dev-lexical'%source)
    print 'training features = ', len([word for sent in train_sentences for word in sent])
    test_sentences = pos_data.load_preprocess_obj('%s-test'%source)
    test_feats = classify_pos.load_classify_obj('%s-test-lexical'%source)
    print 'test features = ', len([word for sent in test_sentences for word in sent])
    feat_list = set([word for sent in train_sentences for word in sent])&set([word for sent in test_sentences for word in sent])
    tag_list = list(set(pos_data.tag_list(train_sentences))&set(pos_data.tag_list(test_sentences)))
    print 'number of tags = ',len(tag_list)
    for nSent,sent in enumerate(train_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = train_feats[nSent][nWord]
                for ft in x:
                    if ft[1] != 0:
                        featFile.write('%d:%f ' % (feat_list.index(ft[0]),ft[1])) 
                featFile.write('\n')
    featFile.close()
    featFile = open(testFileName, 'w')
    
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = test_feats[nSent][nWord]
                for ft in x:
                    if ft[1] != 0:
                        featFile.write('%d:%f ' % (feat_list.index(ft[0]),ft[1])) 
                featFile.write('\n')
    featFile.close()
    # Train using classias.
    modelFileName = '../work/%s/model_lexical.ID' % (source)
    print 'Training...'
    trainMultiLBFGS(trainFileName, modelFileName)
    # Test using classias.
    print 'Testing...'
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print 'Accuracy =', acc
    print 'Intervals=', intervals
    print '###########################################\n\n'
    return acc,intervals
    pass


def evaluate_POS_pivots(source,target,method,n):
    features = pos_data.load_obj(source,target,method) if 'landmark' not in method else pos_data.load_obj(source,target,'/test/'+method)
    pivots = dict(features[:n]).keys()
    print 'pivots for %s-%s, %s top-%d' % (source,target,method,n)
    trainFileName = '../work/%s/%s-%s/trainVects.PV' % (method,source, target)
    testFileName = '../work/%s/%s-%s/testVects.PV' % (method,source, target)

    featFile = open(trainFileName, 'w')
    train_sentences = pos_data.load_preprocess_obj('%s-labeled'%source)
    train_feats = classify_pos.load_classify_obj('%s-labeled-lexical'%source)
    print 'training features = ', len([word for sent in train_sentences for word in sent])
    test_sentences = pos_data.load_preprocess_obj('%s-test'%target)
    test_feats = classify_pos.load_classify_obj('%s-test-lexical'%target)
    print 'test features = ', len([word for sent in test_sentences for word in sent])
    feat_list = set([word for sent in train_sentences for word in sent])&set([word for sent in test_sentences for word in sent])
    tag_list = list(set(pos_data.tag_list(train_sentences))&set(pos_data.tag_list(test_sentences)))
    print 'number of tags = ',len(tag_list)
    for nSent,sent in enumerate(train_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:               
                x = train_feats[nSent][nWord]
                if x[len(x)/2][0] in pivots:
                    featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                    # print x[len(x)/2][0], 'is a pivot!'
                    for ft in x:
                        if ft[1] != 0: #features.get(ft[0],0)
                            featFile.write('%d:%f ' % (feat_list.index(ft[0]),ft[1])) 
                    featFile.write('\n')
    featFile.close()
    featFile = open(testFileName, 'w')
    
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            if pos_tag in tag_list:
                featFile.write('%d '%pos_data.tag_to_number(pos_tag,tag_list))
                x = test_feats[nSent][nWord]
                for ft in x:
                    if ft[1] != 0:
                        featFile.write('%d:%f ' % (feat_list.index(ft[0]),ft[1])) 
                featFile.write('\n')
    featFile.close()
    modelFileName = '../work/%s/%s-%s/model.PV' % (source,target)
    print 'Training...'
    trainMultiLBFGS(trainFileName, modelFileName)
    # Test using classias.
    print 'Testing...'
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print 'Accuracy =', acc
    print 'Intervals=', intervals
    print '###########################################\n\n'
    return acc,intervals
    pass

def batchEval_ID():
    '''
    Evaluate on all 3 domains.
    '''
    resFile = open('../work/batchID.csv', 'w')
    resFile.write('Source, Target, Method, Acc, IntLow, IntHigh\n')
    # source = 'wsj'
    domains = ['answers','reviews','newsgroups']
    for target in domains:
        # evaluate_POS_ID(target)
        source = target
        evaluation = evaluate_POS_ID(target)
        # test_ID(target)
        resFile.write('%s, %s, %s, %f, %f, %f\n' % (source, target, 'ID', evaluation[0], evaluation[1][0],evaluation[1][1]))
        resFile.flush()
    resFile.close()
    pass

def batchEval_ID_lexical():
    '''
    Evaluate on all 3 domains.
    '''
    resFile = open('../work/batchID_lexical.csv', 'w')
    resFile.write('Source, Target, Method, Acc, IntLow, IntHigh\n')
    # source = 'wsj'
    domains = ['answers','reviews','newsgroups']
    for target in domains:
        source = target
        evaluation = evaluate_POS_ID_lexical(target)
        resFile.write('%s, %s, %s, %f, %f, %f\n' % (source, target, 'ID_lexical', evaluation[0], evaluation[1][0],evaluation[1][1]))
        resFile.flush()
    resFile.close()
    pass

def batchEval_NA():
    '''
    Evaluate on all 5 domain pairs. 
    '''
    resFile = open('../work/batchNA.csv', 'w')
    resFile.write('Source, Target, Method, Acc, IntLow, IntHigh\n')
    source = 'wsj'
    # domains = ['answers','emails']
    # domains += ['weblogs','newsgroups']
    domains = ['reviews']
    for target in domains:
        evaluation = evaluate_POS_NA(source, target)
        resFile.write('%s, %s, %s, %f, %f, %f\n' % (source, target, 'NA', evaluation[0], evaluation[1][0],evaluation[1][1]))
        resFile.flush()
    resFile.close()
    pass

def batchEval_NA_lexical():
    '''
    Evaluate on all 5 domain pairs. 
    '''
    resFile = open('../work/batchNA_lexical.csv', 'w')
    resFile.write('Source, Target, Method, Acc, IntLow, IntHigh\n')
    source = 'wsj'
    domains = ['answers','emails']
    domains += ['reviews','newsgroups','weblogs']
    for target in domains:
        evaluation = evaluate_POS_NA_lexical(source, target)
        resFile.write('%s, %s, %s, %f, %f, %f\n' % (source, target, 'NA_lexical', evaluation[0], evaluation[1][0],evaluation[1][1]))
        resFile.flush()
    resFile.close()
    pass

def batchEval(method, gamma, n):
    '''
    Evaluate on all 5 domain pairs. 
    '''
    resFile = open('../work/batchSCL.%s.csv'% method, 'w')
    resFile.write('Source, Target, Method, Acc, IntLow, IntHigh\n')
    source = 'wsj'
    domains = ['answers','emails']
    domains += ['reviews','newsgroups','weblogs']
    for target in domains:
        learnProjection(source, target, method, n)
        evaluation = evaluate_POS(source, target, True, gamma, method, n)
        # evaluation = test_results(source,target,method)
        resFile.write('%s, %s, %s, %f, %f, %f\n' % (source, target, method, evaluation[0], evaluation[1][0],evaluation[1][1]))
        resFile.flush()
    resFile.close()
    pass

def batchEval_lexical(method, gamma, n):
    '''
    Evaluate on all 5 domain pairs. 
    '''
    resFile = open('../work/sim/batchSCL_lexical.%s.csv'% method, 'w')
    resFile.write('Source, Target, Method, Acc, IntLow, IntHigh\n')
    source = 'wsj'
    domains = ['answers','emails']
    domains += ['reviews','newsgroups','weblogs']
    for target in domains:
        # learnProjection(source, target, method, n)
        evaluation = evaluate_POS_lexical(source, target, True, gamma, method, n)
        resFile.write('%s, %s, %s, %f, %f, %f\n' % (source, target, method, evaluation[0], evaluation[1][0],evaluation[1][1]))
        resFile.flush()
    resFile.close()
    pass

def choose_gamma_one_domain_pair(source,target,method,gammas,n):
    resFile = open('../work/a_sim/SCLgamma%s-%s.%s.csv'% (source, target, method), 'w')
    resFile.write('Source, Target, Model, Acc, IntLow, IntHigh, #pivots, gamma\n')
    learnProjection(source, target, method, n)
    for gamma in gammas:    
        evaluation = evaluate_POS(source, target, True, gamma, method, n)
        resFile.write('%s, %s, %s, %f, %f, %f, %f,%f\n' % (source, target, 'combined' , evaluation[0], evaluation[1][0],evaluation[1][1],n,gamma))
        resFile.flush()
    resFile.close()
    pass

def choose_param(method,params,gamma,n):
    resFile = open('../work/sim/SCLparams.%s.csv'% method, 'w')
    resFile.write('Source, Target, Model, Acc, IntLow, IntHigh, Param\n')
    source = 'wsj'
    domains = ['answers','emails']
    domains += ['reviews','newsgroups','weblogs']
    for param in params:
        test_method = 'test_%s_%f'% (method,param)
        for target in domains:
            # learnProjection(source, target, test_method, n)
            evaluation = evaluate_POS(source, target, True, gamma, test_method, n)
            # evaluation = evaluate_POS_lexical(source, target, True, gamma, test_method, n)
            resFile.write('%s, %s, %s, %f, %f, %f, %f\n' % (source, target, method , evaluation[0], evaluation[1][0],evaluation[1][1],param))
            resFile.flush()
    resFile.close()
    pass


# for one domain pair, for each pivot selection method,
# evaluate explicit, implicit and explicit+implicit
def batchEval_one_domain_pair(source,target,method,gamma,n):
    resFile = open('../work/a_sim/SCL%s-%s.%s.csv'% (source,target, method), 'w')
    resFile.write('Source, Target, Model, Acc, IntLow, IntHigh, #pivots\n')
    learnProjection(source, target, method, n)
    # evaluation = evaluate_POS_lexical(source, target, True, gamma, method, n)
    # resFile.write('%s, %s, %s, %f, %f, %f, %f\n' % (source, target, 'explicit' , evaluation[0], evaluation[1][0],evaluation[1][1],n))
    # resFile.flush()
    # evaluation = evaluate_POS_NA(source, target)
    # resFile.write('%s, %s, %s, %f, %f, %f, %f\n' % (source, target, 'implicit' , evaluation[0], evaluation[1][0],evaluation[1][1],n))
    # resFile.flush()
    evaluation = evaluate_POS(source, target, True, gamma, method, n)
    resFile.write('%s, %s, %s, %f, %f, %f, %f\n' % (source, target, 'combined' , evaluation[0], evaluation[1][0],evaluation[1][1],n))
    resFile.flush()
    resFile.close()
    pass




### test methods ###
def test_results(source,target,method,gamma):
    trainFileName = '../work/%s-%s/trainVects.%s' % (source, target,method)
    testFileName = '../work/%s-%s/testVects.%s' % (source, target,method)
    modelFileName = '../work/%s-%s/model.%s' % (source, target,method)
    if 'NA' not in method:
        trainFileName = '../work/%s/%s-%s/trainVects.SCL' % (method,source, target)
        testFileName = '../work/%s/%s-%s/testVects.SCL' % (method,source, target)
        modelFileName = '../work/%s/%s-%s/model.SCL.%f' % (method,source, target,gamma)
    # print 'Training...'
    # trainMultiLBFGS(trainFileName, modelFileName)
    print 'Testing...'
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print 'Accuracy =', acc
    print 'Intervals=', intervals
    print '###########################################\n\n'
    return acc,intervals
    pass

def test_ID(source):
    trainFileName = '../work/%s/trainVects.ID' % (source)
    testFileName = '../work/%s/testVects.ID' % (source)
    modelFileName = '../work/%s/model.ID' % (source)
    # print 'Training...'
    # trainMultiLBFGS(trainFileName, modelFileName)
    print 'Testing...'
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print 'Accuracy =', acc
    print 'Intervals=', intervals
    print '###########################################\n\n'
    return acc,intervals
    pass

# only for labelled methods
# evaluation on balanced score function, basically modify the directory
def dist_evaluate_one_domain_pair(source,target,method,gamma,n):
    # create dir to store scores for different settings
    dist_method = "dist/"+method
    # others remain the same
    resFile = open('../work/dist_sim/SCLdist%s-%s.%s.csv'% (source,target,method), 'w')
    resFile.write('Source, Target, Model, Acc, IntLow, IntHigh, #pivots\n')
    learnProjection(source, target, dist_method, n)
    # explicit: SCL pivot predictors 
    evaluation = evaluate_POS_lexical(source, target, True, gamma, dist_method, n)
    resFile.write('%s, %s, %s, %f, %f, %f, %f\n' % (source, target, 'explicit' , evaluation[0], evaluation[1][0],evaluation[1][1],n))
    resFile.flush()
    # implicit: word embeddings
    evaluation = evaluate_POS_NA(source, target)
    resFile.write('%s, %s, %s, %f, %f, %f, %f\n' % (source, target, 'implicit' , evaluation[0], evaluation[1][0],evaluation[1][1],n))
    resFile.flush()
    # combined = explicit + implicit
    evaluation = evaluate_POS(source, target, True, gamma, dist_method, n)
    resFile.write('%s, %s, %s, %f, %f, %f, %f\n' % (source, target, 'combined' , evaluation[0], evaluation[1][0],evaluation[1][1],n))
    resFile.flush()
    resFile.close()
    pass

def dist_choose_gamma_one_domain_pair(source,target,method,gammas,n):
    dist_method = "dist/"+method
    resFile = open('../work/dist_sim/SCLdistgamma%s-%s.%s.csv'% (source, target, method), 'w')
    resFile.write('Source, Target, Model, Acc, IntLow, IntHigh, #pivots, gamma\n')
    learnProjection(source, target, dist_method, n)
    for gamma in gammas:    
        evaluation = evaluate_POS(source, target, True, gamma, dist_method, n)
        resFile.write('%s, %s, %s, %f, %f, %f, %f,%f\n' % (source, target, 'combined' , evaluation[0], evaluation[1][0],evaluation[1][1],n,gamma))
        resFile.flush()
    resFile.close()
    pass

# f1 
def f1_evaluate_one_domain_pair(source,target,method,gamma,n,opt):
    # create dir to store scores for different settings
    f1_method = ("f1/%s/"%opt)+method
    # others remain the same
    resFile = open('../work/f1_sim/%s/SCLf1%s-%s.%s.csv'% (opt,source,target,method), 'w')
    resFile.write('Source, Target, Model, Acc, IntLow, IntHigh, #pivots\n')
    learnProjection(source, target, f1_method, n)
    # explicit: SCL pivot predictors 
    evaluation = evaluate_POS_lexical(source, target, True, gamma, f1_method, n)
    resFile.write('%s, %s, %s, %f, %f, %f, %f\n' % (source, target, 'explicit' , evaluation[0], evaluation[1][0],evaluation[1][1],n))
    resFile.flush()
    # implicit: word embeddings
    evaluation = evaluate_POS_NA(source, target)
    resFile.write('%s, %s, %s, %f, %f, %f, %f\n' % (source, target, 'implicit' , evaluation[0], evaluation[1][0],evaluation[1][1],n))
    resFile.flush()
    # combined = explicit + implicit
    evaluation = evaluate_POS(source, target, True, gamma, f1_method, n)
    resFile.write('%s, %s, %s, %f, %f, %f, %f\n' % (source, target, 'combined' , evaluation[0], evaluation[1][0],evaluation[1][1],n))
    resFile.flush()
    resFile.close()
    pass

def f1_choose_gamma_one_domain_pair(source,target,method,gammas,n,opt):
    f1_method = ("f1/%s/"%opt)+method
    resFile = open('../work/f1_sim/%s/SCLf1gamma%s-%s.%s.csv'% (opt, source, target, method), 'w')
    resFile.write('Source, Target, Model, Acc, IntLow, IntHigh, #pivots, gamma\n')
    learnProjection(source, target, f1_method, n)
    for gamma in gammas:    
        evaluation = evaluate_POS(source, target, True, gamma, f1_method, n)
        resFile.write('%s, %s, %s, %f, %f, %f, %f,%f\n' % (source, target, 'combined' , evaluation[0], evaluation[1][0],evaluation[1][1],n,gamma))
        resFile.flush()
    resFile.close()
    pass


# number of pivots
def evaluate_numbers_of_pivots(source,target,method,gamma,nums):

    pass


if __name__ == '__main__':
    source = 'wsj'
    target = 'answers'
    # target = 'reviews'
    method = 'ppmi'
    # method = 'un_mi'
    # method = "un_mi"
    # methods = ['mi','un_mi','pmi','un_pmi','freq','un_freq','mi','un_mi','ppmi','un_ppmi']
    n = 500
    
    # batchEval(method, 1, n)
    # batchEval_NA()
    # learnProjection(source, target, method, n)
    # evaluate_POS_lexical(source, target, True, 1, method, n)
    evaluate_POS(source, target, True, 1 ,method, n)
    evaluate_POS_NA(source,target)
    # evaluate_POS_NA_lexical(source,target)
    # test_results(source,target,method,1)
    # evaluate_POS_ID(target)
    # evaluate_POS_pivots(source,target,method,n)
    # batchEval_ID()
    # test_ID(target)
    # batchEval_ID_lexical()
    # batchEval_NA_lexical()
    # evaluate_POS_ID_lexical(target)
    # methods = ['un_mi','pmi']
    # pos_tag = 'NN'
    # methods = ['%s.%s'%(method,pos_tag)]

    # methods = ['freq','un_freq','mi','un_mi','ppmi','un_ppmi']
    # methods = ['freq','mi','pmi','ppmi']
    # opt = 'r'
    # opt = 'w'
    # methods = ['pmi','un_pmi','freq','un_freq','mi','un_mi','ppmi','un_ppmi']
    # methods += ['landmark_pretrained_word2vec','landmark_pretrained_word2vec_ppmi','landmark_pretrained_glove','landmark_pretrained_glove_ppmi']
    # methods = ['landmark_pretrained_word2vec','landmark_pretrained_glove']
    # methods = ['pmi','un_pmi','ppmi','un_ppmi']
    # for method in methods:
    #     batchEval(method, 1, n)
        # batchEval_one_domain_pair(source,target,method,1,n)
        # batchEval_lexical(method, 1, n)
        # dist_evaluate_one_domain_pair(source,target,method,1,n)
        # f1_evaluate_one_domain_pair(source,target,method,1,n,opt)
    # gammas = [0.01,0.1,1,10,100]
    # for method in methods:
        # dist_choose_gamma_one_domain_pair(source, target, method,gammas,n)
        # choose_gamma_one_domain_pair(source, target, method,gammas,n)
        # f1_choose_gamma_one_domain_pair(source, target, method,gammas,n,opt)
    # params = [1]
    # params = [0,0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
    # params += [10e-3,10e-4,10e-5,10e-6]
    # params.sort()
    # params = [1,50,100,1000,10000]
    # params = [0,1,50,100,1000,10000]
    # for method in methods:
    #     choose_param(method,params,1,n)

