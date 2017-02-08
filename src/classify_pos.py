import landmark_pivot as lp
import numpy
import os
import glob
import pickle
import pos_data
# import gensim,logging
# from glove import Corpus, Glove
# import glove


#### prepare classification data ####
# words to lexical features by a window size 2l+1
# default l = 2
def window_features(name,sentences,l):
    new_sentences= []
    for sent in sentences:
        new_sent = []
        freq_sent={}
        get_freq_in_sentence(sent,freq_sent)
        for word in sent:
            new_word = []
            for i in range(-l,l+1):
                word_postion = word[2]+i
                word_i = find_word_in_position(sent,word_postion)
                new_word.append([word_i,freq_sent.get(word_i,0)])
                # print len(new_word)
            new_sent += [new_word]
        # print len(sent),len(new_sent)
        new_sentences+= [new_sent]
    # print len(sentences), len(new_sentences)
    save_classify_obj(new_sentences,'%s-lexical'%name)
    pass

def get_freq_in_sentence(sent,h):
    for word in sent:
        h[word[0]]=h.get(word[0],0)+1
    pass

#### prepare classification data ####
# words to word vectors by a window size 2l+1
# default l = 2
def window_vectors(name,sentences,l):
    # [word]: 0 word, 1 tag, 2 position, 3 sentence_length
    # new_sentences = [[word[0] for word in sent] for sent in sentences]
    path = '../data/glove.42B.300d.txt'
    model = load_filtered_glove(sentences,path)
    print "Loading self-trained model"
    ds_model = Glove.load('../work/classify/glove-selftrained.model')

    new_sentences = []
    for sent in sentences:
        new_sent = []
        for word in sent:
            new_word = []
            for i in range(-l,l+1):
                word_postion = word[2]+i
                word_i = find_word_in_position(sent,word_postion)
                word_vector = word_to_300d(ds_model,model,word_i)
                new_word = joint_vectors(new_word,word_vector)
                # print len(new_word)
            new_sent += [new_word]
        # print len(sent),len(new_sent)
        new_sentences+= [new_sent]
    # print len(sentences), len(new_sentences)
    save_classify_obj(new_sentences,'%s-classify'%name)
    pass

# load GloVe pretrained embeddings or self trained embeddings or zeros
def word_to_300d(ds_model,model,x):
    if x == 0:
        # emtpy word with zeros
        return numpy.zeros(300, dtype=float)
    else:
        if model.get(x,0)==0:
            # print x
            if x not in ds_model.dictionary:
                # print "zeros"
                return numpy.zeros(300, dtype=float)
            else:
                return lp.glove_to_vec(x,ds_model)
        else:
            return lp.word_to_vec(x,model)
    pass

def find_word_in_position(sent,position):
    # print position
    for word in sent:
        if sent.index(word)+1==position:
            # print word[0]
            return word[0]
#     print"zero"
    return 0

def joint_vectors(a,b):
    return numpy.concatenate((a,b))

def load_filtered_glove(sentences,gloveFile):
    print "Loading pretrained GloVe Model"
    f = open(gloveFile,'r')
    model = {}
    filtered_features = lp.pos_data.feature_list(sentences)
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

def glove_single(domain_name):
    corpus_model = Corpus()
    corpus_model.fit(lp.labeled_sentences(domain_name), window=10)
    corpus_model.save('../work/classify/corpus-selftrained.model')
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    print('Training the GloVe model')
    model = Glove(no_components=300, learning_rate=0.05)
    model.fit(corpus_model.matrix, epochs=int(10),
              no_threads=6, verbose=True)
    model.add_dictionary(corpus_model.dictionary)
    model.save('../work/classify/glove-selftrained.model') 
    return

def glove_group():
    corpus_model = Corpus()
    sentences = lp.labeled_sentences("wsj")
    domains = ["answers","emails"]
    domains += ["reviews","newsgroups","weblogs"]
    for domain in domains:
        print "current domain: ", domain
        sentences+=lp.labeled_sentences_test(domain)
    domains = ["answers","reviews","newsgroups"]
    for domain in domains:
        print "current domain: ", domain
        sentences+=lp.labeled_sentences_dev(domain)
    corpus_model.fit(sentences, window=10)
    corpus_model.save('../work/classify/corpus-selftrained.model')
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    print('Training the GloVe model')
    model = Glove(no_components=300, learning_rate=0.05)
    model.fit(corpus_model.matrix, epochs=int(10),
              no_threads=6, verbose=True)
    model.add_dictionary(corpus_model.dictionary)
    model.save('../work/classify/glove-selftrained.model') 
    return

######## prepare test data #########
# training data was prepared in pos_data, however for the classification,
# we also need to divide test data into groups by pos_tag
def divide_test_data(source,target):
    tgt_test = pos_data.load_preprocess_obj('%s-test'%target)
    src_labeled = pos_data.load_preprocess_obj('%s-labeled'%source)
    tags = pos_data.tag_list(src_labeled)

    for pos_tag in tags:
        print "TAG = %s"% pos_tag
        divide_test_data_tag(target,pos_tag,tgt_test)
    pass

def divide_test_data_tag(target,pos_tag,tgt_test):
    # list sentences HAS pos_tag
    pos_tgt_data = pos_data.sentence_list_contain_tag(pos_tag,tgt_test)
    # list sentences NOT pos_tag
    neg_tgt_data = pos_data.minus_lists(tgt_test,pos_tgt_data)
    # save test objects to target domain folders
    pos_tag = 'TAG.' if pos_tag == '.' else pos_tag
    save_test_obj(target,pos_tgt_data,pos_tag,"pos_tgt_data")
    save_test_obj(target,neg_tgt_data,pos_tag,"neg_tgt_data")
    pass


# save and load for classification
def save_classify_obj(obj, name):
    filename = '../work/classify/'+name + '.pkl'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_classify_obj(name):
    with open('../work/classify/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)

# save and load for testing
def save_test_obj(target,obj,tag,name):
    filename = '../work/%s/%s/%s.pkl'%(target,tag,name)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print "%s saved"%filename

def load_test_obj(target,tag,name):
    with open('../work/%s/%s/%s.pkl'%(target,tag,name), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    l = 2
    # name = 'answers-test'
    # sentences=lp.pos_data.load_preprocess_obj(name)
    # window_vectors(name,sentences,l)
    # window_features(name,sentences,l)
    my_dir = '../work/preprocess'
    names = [name.replace('.pkl','') for name in os.listdir(my_dir)]
    for name in names:
        if 'unlabeled' not in name:
            print name
            sentences=lp.pos_data.load_preprocess_obj(name)
            # window_vectors(name,sentences,l)
            window_features(name,sentences,l)
        # if 'unlabeled' in name:
        #     print name
        #     sentences=lp.pos_data.load_preprocess_obj(name)
        #     window_vectors(name,sentences,l)

    # source = 'wsj'
    # glove_single(source)
    # glove_group()
    # domains = ["answers","emails"]
    # domains += ["reviews","newsgroups","weblogs"]
    # for target in domains:
    #     divide_test_data(source,target)