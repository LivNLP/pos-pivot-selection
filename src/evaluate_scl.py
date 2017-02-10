"""
Forked Code from Danushka Bollegala
Implementation of SCL following steps after pivot selection
Used for evaluation of pivot selection methods
-----------

Change log: make some changes to do a multilabel classification
"""

import numpy as np
import scipy.io as sio 
import scipy.sparse as sp
from sparsesvd import sparsesvd

import sys, math, subprocess, time

import pos_data
import classify_pos
import re
import scipy.stats

def clopper_pearson(k,n,alpha=0.05):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    """
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi

def trainLBFGS(train_file, model_file):
    """
    Train lbfgs on train file. and evaluate on test file.
    Read the output file and return the classification accuracy.
    """
    retcode = subprocess.call(
        "classias-train -tb -a lbfgs.logistic -pc1=0 -pc2=1 -m %s %s > /dev/null"  %\
        (model_file, train_file), shell=True)
    return retcode


def trainMultiLBFGS(train_file, model_file):
    """
    Train lbfgs on train file. and evaluate on test file. different from the previous one!
    Read the output file and return the multi-label classification accuracy.
    """
    retcode = subprocess.call(
        "classias-train -tn -a lbfgs.logistic -pc1=0 -pc2=1 -m %s %s > /dev/null"  %\
        (model_file, train_file), shell=True)
    return retcode


def testLBFGS(test_file, model_file):
    """
    Evaluate on the test file.
    Read the output file and return the classification accuracy.
    """
    output = "../work/output"
    retcode = subprocess.call("cat %s | classias-tag -m %s -t -fap > %s" %\
                              (test_file, model_file, output), shell=True)
    F = open(output)
    accuracy = 0
    correct = 0
    total = 0
    for line in F:
        if line.startswith("Accuracy"):
            p = line.strip().split()
            accuracy = float(p[1])
            [correct, total]=[int(s) for s in re.findall(r'\b\d+\b',p[2])]
    F.close()
    return accuracy,correct,total


def loadClassificationModel(modelFileName):
    """
    Read the model file and return a list of (feature, weight) tuples.
    """
    modelFile = open(modelFileName, "r") 
    weights = []
    for line in modelFile:
        if line.startswith('@'):
            # this is @classias or @bias. skip those.
            continue
        p = line.strip().split()
        featName = p[1].strip()
        featVal = float(p[0])
        if featName == "__BIAS__":
            # This is the bias term
            bias = featVal
        else:
            # This is an original feature.
            if featVal > 0:
                weights.append((featName, featVal))
    modelFile.close()
    return weights

def selectTh(h, t):
    """
    Select all elements of the dictionary h with frequency greater than t. 
    """
    p = {}
    for (key, val) in h.iteritems():
        if val > t:
            p[key] = val
    del(h)
    return p


def learnProjection(sourceDomain, targetDomain, pivotsMethod, n):
    """
    Learn the projection matrix and store it to a file. 
    """
    h = 50 # no. of SVD dimensions.
    #n = 500 # no. of pivots.

    # Parameters to reduce the number of features in the tail
    domainTh = {'wsj':5, 'answers':5, 'emails':5, 'reviews':5, 'weblogs':5,'newsgroups':5}

    # Load pivots.
    features = pos_data.load_obj(sourceDomain,targetDomain,pivotsMethod)
    pivots = dict(features[:n]).keys()
    print "selecting top-%d features in %s as pivots" % (n, pivotsMethod)

    # Load features and get domain specific features
    features = pos_data.load_obj(sourceDomain,targetDomain,"un_freq") if "un_" in pivotsMethod else pos_data.load_obj(sourceDomain,targetDomain,"freq")
    feats = selectTh(dict(features),domainTh[sourceDomain])
    if "landmark" in pivotsMethod:
        feats = pos_data.load_obj(sourceDomain,targetDomain,"filtered_features")
    print "experimental features = ", len(feats)
    # print feats.keys()

    # DSwords = [item for item in feats if item not in pivots]

    feats = feats.keys()
    # Load train vectors.
    print "Loading Training vectors...",
    startTime = time.time()
    vects = []
    vects.extend(loadFeatureVecors(pos_data.load_preprocess_obj('%s-labeled'%sourceDomain), feats))
    vects.extend(loadFeatureVecors(pos_data.load_preprocess_obj('%s-unlabeled'%sourceDomain), feats))
    vects.extend(loadFeatureVecors(pos_data.load_preprocess_obj('%s-unlabeled'%targetDomain), feats))
    endTime = time.time()
    print "%ss" % str(round(endTime-startTime, 2))     

    print "Total no. of documents =", len(vects)
    print "Total no. of features =", len(feats)

    # Learn pivot predictors.
    print "Learning Pivot Predictors.."
    startTime = time.time()
    M = sp.lil_matrix((len(feats), len(pivots)), dtype=np.float)
    for (j, w) in enumerate(pivots):
        print "%d of %d %s" % (j, len(pivots), w)
        for (feat, val) in getWeightVector(w, vects):
            i = feats.index(feat)
            M[i,j] = val
    endTime = time.time()
    print "Took %ss" % str(round(endTime-startTime, 2))   

    # Perform SVD on M
    # print M[0].shape
    print "Perform SVD on the weight matrix...",
    startTime = time.time()
    ut, s, vt = sparsesvd(M.tocsc(), h)
    endTime = time.time()
    print "%ss" % str(round(endTime-startTime, 2))     
    sio.savemat("../work/%s-%s/proj.mat" % (sourceDomain, targetDomain), {'proj':ut.T})
    pass


def getWeightVector(word, vects):
    """
    Train a binary classifier to predict the given word and 
    return the corresponding weight vector. Not depending on the task,
    always a binary classification, a pivot or not a pivot.
    """
    trainFileName = "../work/trainFile"
    modelFileName = "../work/modelFile"
    trainFile = open(trainFileName, 'w')
    for v in vects:
        fv = v.copy()
        if word in fv:
            label = 1
            fv.remove(word)
        else:
            label = -1
        trainFile.write("%d %s\n" % (label, " ".join(fv)))
    trainFile.close()
    trainLBFGS(trainFileName, modelFileName)
    return loadClassificationModel(modelFileName)


def loadFeatureVecors(sentences, feats):
    """
    Returns a list of lists that contain features for a document. 
    """
    # sentences = pos_data.format_sentences(old_sentences)
    L = []
    for sent in sentences:
        L.append(set([word[0] for word in sent])&(set(feats)))
    # print L
    return L


def evaluate_POS(source, target, project, gamma, method, n):
    """
    Report the cross-domain POS classification accuracy. 
    """
    # Parameters to reduce the number of features in the tail
    domainTh = {'wsj':5, 'answers':5, 'emails':5, 'reviews':5, 'weblogs':5,'newsgroups':5}

    # gamma = 1.0
    nEmbed = 1500
    nLex = 5
    print "Source Domain", source
    print "Target Domain", target
    if project:
        print "Projection ON", "Gamma = %f" % gamma
    else:
        print "Projection OFF"
    # Load the projection matrix.
    M = sp.csr_matrix(sio.loadmat("../work/%s-%s/proj.mat" % (source, target))['proj'])
    (nDS, h) = M.shape

    # Load pivots.
    features = pos_data.load_obj(source,target,method)
    pivots = dict(features[:n]).keys()
    print "selecting top-%d features in %s as pivots" % (n, method)

    # Load features
    features = pos_data.load_obj(source,target,"un_freq") if "un_" in method else pos_data.load_obj(source,target,"freq")
    feats = selectTh(dict(features),domainTh[source])
    if "landmark" in method:
        feats = pos_data.load_obj(source,target,"filtered_features")
    print "experimental features = ", len(feats)
    #print feats
    # DSwords = [item for item in feats if item not in pivots]

    feats = feats.keys()
    # write train feature vectors.
    trainFileName = "../work/%s-%s/trainVects.SCL" % (source, target)
    testFileName = "../work/%s-%s/testVects.SCL" % (source, target)
    featFile = open(trainFileName, 'w')
    
    train_sentences = pos_data.load_preprocess_obj("%s-labeled"%source)
    train_vectors = classify_pos.load_classify_obj("%s-labeled-classify"%source)
    # load lexical features as additional features
    train_feats = classify_pos.load_classify_obj("%s-labeled-lexical"%source)
    for nSent,sent in enumerate(train_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            featFile.write("%d "%pos_data.tag_to_number(pos_tag))
            x = sp.lil_matrix((1, nDS), dtype=np.float64)
            x[0,:nEmbed] = train_vectors[nSent][nWord]
            # x[0,nEmbed:nEmbed+nLex] = train_feats[nSent][nWord]
            if project:
                y = x.tocsr().dot(M)
                for i in range(0, h):
                    featFile.write("proj_%d:%f " % (i, gamma * y[0,i])) 
            lex = train_feats[nSent][nWord]
            for ft in lex:
                featFile.write("%s:%f " % (ft[0],ft[1])) 
            featFile.write("\n")
    featFile.close()
    featFile = open(testFileName, 'w')
    test_sentences = pos_data.load_preprocess_obj("%s-test"%target)
    test_vectors = classify_pos.load_classify_obj("%s-test-classify"%target)
    # load lexical features as additional features
    test_feats = classify_pos.load_classify_obj("%s-test-lexical"%target)
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            featFile.write("%d "%pos_data.tag_to_number(pos_tag))
            x = sp.lil_matrix((1, nDS), dtype=np.float64)
            x[0,:nEmbed] = test_vectors[nSent][nWord]
            if project:
                y = x.tocsr().dot(M)
                for i in range(0, h):
                    featFile.write("proj_%d:%f " % (i, gamma * y[0,i])) 
            lex = test_feats[nSent][nWord]
            for ft in lex:
                featFile.write("%s:%f " % (ft[0],ft[1])) 
            featFile.write("\n")
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s-%s/model.SCL" % (source, target)
    trainMultiLBFGS(trainFileName, modelFileName)
    # Test using classias.
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print "Accuracy =", acc
    print "Intervals=", intervals
    print "###########################################\n\n"
    return acc,intervals

def evaluate_POS_NA(source,target):
    trainFileName = "../work/%s-%s/trainVects.NA" % (source, target)
    testFileName = "../work/%s-%s/testVects.NA" % (source, target)
    featFile = open(trainFileName, 'w')
    count = 0
    train_sentences = pos_data.load_preprocess_obj("%s-labeled"%source)
    train_vectors = classify_pos.load_classify_obj("%s-labeled-classify"%source)
    # load lexical features as additional features
    # train_feats = classify_pos.load_classify_obj("%s-labeled-lexical"%source)
    print "training features = ", len(pos_data.feature_list(train_sentences))
    for nSent,sent in enumerate(train_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            featFile.write("%d "%pos_data.tag_to_number(pos_tag))
            x = train_vectors[nSent][nWord]
            for i in x:
                featFile.write("%f " % i) 
            featFile.write("\n")
    featFile.close()
    featFile = open(testFileName, 'w')
    test_sentences = pos_data.load_preprocess_obj("%s-test"%target)
    test_vectors = classify_pos.load_classify_obj("%s-test-classify"%target)
    # load lexical features as additional features
    # test_feats = classify_pos.load_classify_obj("%s-test-lexical"%target)
    print "test features = ", len(pos_data.feature_list(test_sentences))
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            featFile.write("%d "%pos_data.tag_to_number(pos_tag))
            x = test_vectors[nSent][nWord]
            for i in x:
                featFile.write("%f " % i) 
            featFile.write("\n")
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s-%s/model.NA" % (source, target)
    print "Training..."
    trainMultiLBFGS(trainFileName, modelFileName)
    # Test using classias.
    print "Testing..."
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print "Accuracy =", acc
    print "Intervals=", intervals
    print "###########################################\n\n"
    return acc,intervals
    pass

def evaluate_POS_NA_lexical(source,target):
    trainFileName = "../work/%s-%s/trainVects_lexical.NA" % (source, target)
    testFileName = "../work/%s-%s/testVects_lexical.NA" % (source, target)
    featFile = open(trainFileName, 'w')
    count = 0
    train_sentences = pos_data.load_preprocess_obj("%s-labeled"%source)
    train_feats = classify_pos.load_classify_obj("%s-labeled-lexical"%source)
    print "training features = ", len(pos_data.feature_list(train_sentences))
    for nSent,sent in enumerate(train_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            featFile.write("%d "%pos_data.tag_to_number(pos_tag))
            x = train_feats[nSent][nWord]
            # print x
            for ft in x:
                featFile.write("%s:%f " % (ft[0],ft[1])) 
            featFile.write("\n")
    featFile.close()
    featFile = open(testFileName, 'w')
    test_sentences = pos_data.load_preprocess_obj("%s-test"%target)
    test_feats = classify_pos.load_classify_obj("%s-test-lexical"%target)
    print "test features = ", len(pos_data.feature_list(test_sentences))
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            featFile.write("%d "%pos_data.tag_to_number(pos_tag))
            x = test_feats[nSent][nWord]
            for ft in x:
                featFile.write("%s:%f " % (ft[0],ft[1])) 
            featFile.write("\n")
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s-%s/model_lexical.NA" % (source, target)
    print "Training..."
    trainMultiLBFGS(trainFileName, modelFileName)
    # Test using classias.
    print "Testing..."
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print "Accuracy =", acc
    print "Intervals=", intervals
    print "###########################################\n\n"
    return acc,intervals
    pass



def evaluate_POS_ID(source):
    trainFileName = "../work/%s/trainVects.ID" % (source)
    testFileName = "../work/%s/testVects.ID" % (source)
    featFile = open(trainFileName, 'w')
    count = 0
    print "Loading training vectors..."
    train_sentences = pos_data.load_preprocess_obj("%s-dev"%source)
    train_vectors = classify_pos.load_classify_obj("%s-dev-classify"%source)
    print "training features = ", len(pos_data.feature_list(train_sentences))
    for nSent,sent in enumerate(train_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            featFile.write("%d "%pos_data.tag_to_number(pos_tag))
            x = train_vectors[nSent][nWord]
            for i in x:
                featFile.write("%f " % i) 
            featFile.write("\n")
    featFile.close()
    featFile = open(testFileName, 'w')
    print "Loading test vectors..."
    test_sentences = pos_data.load_preprocess_obj("%s-test"%source)
    test_vectors = classify_pos.load_classify_obj("%s-test-classify"%source)
    print "test features = ", len(pos_data.feature_list(test_sentences))
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            featFile.write("%d "%pos_data.tag_to_number(pos_tag))
            x = test_vectors[nSent][nWord]
            for i in x:
                featFile.write("%f " % i) 
            featFile.write("\n")
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s/model.ID" % (source)
    trainMultiLBFGS(trainFileName, modelFileName)
    # Test using classias.
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print "Accuracy =", acc
    print "Intervals=", intervals
    print "###########################################\n\n"
    return acc,intervals
    pass


def evaluate_POS_ID_lexical(source):
    trainFileName = "../work/%s/trainVects_lexical.ID" % (source)
    testFileName = "../work/%s/testVects_lexical.ID" % (source)
    featFile = open(trainFileName, 'w')
    count = 0
    train_sentences = pos_data.load_preprocess_obj("%s-dev"%source)
    train_feats = classify_pos.load_classify_obj("%s-dev-lexical"%source)
    print "training features = ", len(pos_data.feature_list(train_sentences))
    for nSent,sent in enumerate(train_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            featFile.write("%d "%pos_data.tag_to_number(pos_tag))
            x = train_feats[nSent][nWord]
            # print x
            for ft in x:
                featFile.write("%s:%f " % (ft[0],ft[1])) 
            featFile.write("\n")
    featFile.close()
    featFile = open(testFileName, 'w')
    test_sentences = pos_data.load_preprocess_obj("%s-test"%source)
    test_feats = classify_pos.load_classify_obj("%s-test-lexical"%source)
    print "test features = ", len(pos_data.feature_list(test_sentences))
    for nSent,sent in enumerate(test_sentences):
        words = [word[0] for word in sent]
        for nWord,w in enumerate(words):
            pos_tag = sent[nWord][1]
            featFile.write("%d "%pos_data.tag_to_number(pos_tag))
            x = test_feats[nSent][nWord]
            for ft in x:
                featFile.write("%s:%f " % (ft[0],ft[1])) 
            featFile.write("\n")
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s/model_lexical.ID" % (source)
    print "Training..."
    trainMultiLBFGS(trainFileName, modelFileName)
    # Test using classias.
    print "Testing..."
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print "Accuracy =", acc
    print "Intervals=", intervals
    print "###########################################\n\n"
    return acc,intervals
    pass


def batchEval(method, gamma, n):
    """
    Evaluate on all 5 domain pairs. 
    """
    resFile = open("../work/batchSCL.%s.csv"% method, "w")
    resFile.write("Source, Target, Method, Acc, IntLow, IntHigh\n")
    source = 'wsj'
    domains = ["answers","emails"]
    domains += ["reviews","newsgroups","weblogs"]
    for target in domains:
        learnProjection(source, target, method, n)
        evaluation = evaluate_POS(source, target, True, gamma, method, n)
        resFile.write("%s, %s, %s, %f, %f, %f\n" % (source, target, method, evaluation[0], evaluation[1][0],evaluation[1][1]))
        resFile.flush()
    resFile.close()
    pass


def choose_gamma(source, target, method, gammas, n):
    resFile = open("../work/gamma/%s-%s/SCLgamma.%s.csv"% (source, target, method), "w")
    resFile.write("Source, Target, Method, Proj, Gamma\n")
    learnProjection(source, target, method, n)
    for gamma in gammas:    
        resFile.write("%s, %s, %s, %f, %f\n" % (source, target, method, evaluate_POS(source, target, True, gamma, method, n), gamma))
        resFile.flush()
    resFile.close()
    pass

def choose_param(method,params,gamma,n):
    resFile = open("../work/sim/SCLparams.%s.csv"% method, "w")
    resFile.write("Source, Target, Model, Acc, IntLow, IntHigh, Param\n")
    source = 'wsj'
    domains = ["answers","emails"]
    domains += ["reviews","newsgroups","weblogs"]
    for param in params:
        test_method = "test_%s_%f"% (method,param)
        for target in domains:
            learnProjection(source, target, test_method, n)
            evaluation = evaluate_POS(source, target, True, gamma, test_method, n)
            resFile.write("%s, %s, %s, %f, %f, %f, %f\n" % (source, target, method , evaluation[0], evaluation[1][0],evaluation[1][1],param))
            resFile.flush()
    resFile.close()
    pass

if __name__ == "__main__":
    source = "wsj"
    target = "answers"
    # batchNA()
    # batchID()
    method = "freq"
    learnProjection(source, target, method, 500)
    evaluate_POS(source, target, True, 1,method, 500)
    # evaluate_POS_NA(source,target)
    # evaluate_POS_NA_lexical(source,target)
    # evaluate_POS_ID(target)
    # evaluate_POS_ID_lexical(target)
    # methods = ["freq","un_freq","mi","un_mi","pmi","un_pmi"]
    # methods += ["ppmi",'un_ppmi']
    # methods = ["mi","un_mi","pmi","un_pmi"]
    # methods += ["landmark_pretrained_word2vec","landmark_pretrained_word2vec_ppmi","landmark_pretrained_glove","landmark_pretrained_glove_ppmi"]
    # methods = ["landmark_pretrained_word2vec","landmark_pretrained_glove"]
    # n = 500
    # for method in methods:
    #     batchEval(method, 1, n)
    # gammas = [1,5,10,20,50,100]
    # for method in methods:
        # choose_gamma(source, target, method,gammas,n)
    # params = [0,0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
    # params += [10e-3,10e-4,10e-5,10e-6]
    # params.sort()
    # params = [1,50,100,1000,10000]
    # params = [0,1,50,100,1000,10000]
    # for method in methods:
    #     choose_param(method,params,1,n)
