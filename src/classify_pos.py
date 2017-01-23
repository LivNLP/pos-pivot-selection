import landmark_pivot as lp
import numpy
import os
import glob
import pickle
# counter for unfound vectors
count=0

#### prepare classification data ####
# words to word vectors by a window size 2l+1
# default l = 2
def window_vectors(name,sentences,l):
    # [word]: 0 word, 1 tag, 2 position, 3 sentence_length
    # new_sentences = [[word[0] for word in sent] for sent in sentences]
    path = '../data/glove.42B.300d.txt'
    model = load_filtered_glove(sentences,path)

    new_sentences = []
    for sent in sentences:
        new_sent = []
        for word in sent:
            new_word = []
            for i in range(-l,l+1):
                word_postion = word[2]+i
                word_i = find_word_in_position(sent,word_postion)
                word_vector = word_to_300d(model,word_i)
                new_word = joint_vectors(new_word,word_vector)
                print len(new_word)
            new_sent += [new_word]
        new_sentences+= [new_sent]
    print len(sentences), len(new_sentences)
    save_classify_obj(new_sentences,'%s-classify'%name)
    pass

# load GloVe embeddings or 0s
def word_to_300d(model,word):
    if word==0:
        # emtpy word with zeros
        return numpy.zeros(300, dtype=float)
    else:
        # print word
        if model.get(word,0)==0:
            # different from sentpiv, we use empty for unfound vector
            return numpy.zeros(300, dtype=float)
            count+=1
            print count
        else:
            return lp.word_to_vec(word,model)
    pass

def find_word_in_position(sent,position):
    for word in sent:
        if word[2]==position:
            return word[0]
        else:
            return 0
    pass

def joint_vectors(a,b):
    return numpy.concatenate((a,b))

def load_filtered_glove(sentences,gloveFile):
    print "Loading Glove Model"
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

def load_preprocess_obj(name):
    with open('../work/classify/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)



if __name__ == "__main__":
    l = 2
    # name = 'answers-dev'
    # sentences=lp.pos_data.load_preprocess_obj(name)
    my_dir = '../work/preprocess'
    names = [name.replace('.pkl','') for name in os.listdir(my_dir)]
    for name in names:
        print name
        sentences=lp.pos_data.load_preprocess_obj(name)
        window_vectors(name,sentences,l)