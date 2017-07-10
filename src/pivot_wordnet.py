from nltk.corpus import wordnet
import pos_data
import random

# determine whether given word is noun (1) or not (0)
# def is_noun(word):
#     syns=wordnet.synsets(word)
#     # if the word is in the datasets
#     if syns:
#         return 1 if str(syns[0].name()).split('.')[1]=='n' else 0
#     else:
#         return 0


# we give up to do this by wordnet, but use the annonations in the datasets
def is_noun(word,pos_list):
    return 1 if word in pos_list else 0


# read a list of words 
# to decide how many nouns (distribution) in the selected pivots 
def count_nouns(selected_pivots,pos_list):
    return float(sum(1 for word in selected_pivots if is_noun(word,pos_list)==1))/float(len(selected_pivots))

def runner(source,target,method,n):
    # src_labeled = pos_data.load_preprocess_obj('%s-labeled'%source)
    # pos_list = pos_data.feature_list_contain_tag('NN',src_labeled)
    # print len(pos_list)
    # pos_data.save_preprocess_obj(pos_list,'NN_list')
    pos_list = pos_data.load_preprocess_obj('NN_list')
    features = pos_data.load_obj(source,target,method) if 'landmark' not in method else pos_data.load_obj(source,target,'/test/'+method)
    pivots = dict(features[:n]).keys() if n >0 else dict(features[n:]).keys()
    print count_nouns(pivots,pos_list),n
    return count_nouns(pivots,pos_list)

def random_runner(source,target,method,n):
    sum_nouns=0
    features = pos_data.load_obj(source,target,method) if 'landmark' not in method else pos_data.load_obj(source,target,'/test/'+method)
    pivots_loop = [dict(random.sample(features,n)).keys() for i in range(10)]
    for pivots in pivots_loop:
        print count_nouns(pivots),n
        sum_nouns+=count_nouns(pivots)
    print "average = ", sum_nouns/len(pivots_loop)
    return sum_nouns/len(pivots_loop)

# different methods
def batch_results_from_methods(source,target,methods,n):
    f = open('../work/a_sim/%s-%s_nouns.csv'%(source,target), 'w')
    print "source = ", source
    print "target = ", target
    f.write("Source, Target, Method, Nouns, #pivots\n")
    for method in methods:
        print "method = ", method
        nouns = runner(source,target,method,n)
        f.write("%s, %s, %s, %f, %f\n"%(source,target,method,nouns,n))
        f.flush()
    f.close()
    pass

# different methods focus on nouns
def batch_results_from_NN_methods(source,target,methods,n):
    f = open('../work/a_sim/%s-%s_NN_nouns.csv'%(source,target), 'w')
    print "source = ", source
    print "target = ", target
    f.write("Source, Target, Method, Nouns, #pivots\n")
    for method in methods:
        method = "%s.NN"%method
        print "method = ", method
        nouns = runner(source,target,method,n)
        f.write("%s, %s, %s, %f, %f\n"%(source,target,method,nouns,n))
        f.flush()
    f.close()
    pass

# different methods on f1 score with opt 
def batch_results_from_f1_methods(source,target,methods,n,opt):
    f = open('../work/f1_sim/%s-%s_f1_nouns.%s.csv'%(source,target,opt), 'w')
    print "source = ", source
    print "target = ", target
    f.write("Source, Target, Method, Nouns, #pivots\n")
    for method in methods:
        method = ("f1/%s/"%opt) + method
        print "method = ", method
        nouns = runner(source,target,method,n)
        f.write("%s, %s, %s, %f, %f\n"%(source,target,method,nouns,n))
        f.flush()
    f.close()
    pass

# different dist methods
def batch_dist_results_from_methods(source,target,methods,n):
    f = open('../work/dist_sim/%s-%sdist_nouns.csv'%(source,target), 'w')
    print "source = ", source
    print "target = ", target  
    f.write("Source, Target, Method, Nouns, #pivots\n")
    for method in methods:
        pv_method = "dist/"+method
        print "method = ", method
        nouns = runner(source,target,pv_method,n)
        f.write("%s, %s, %s, %f, %f\n"%(source,target,method,nouns,n))
        f.flush()
    f.close()
    pass

# different number of pivots
def batch_reuslts_from_numbers(source,target,method,nums):
    f = open('../work/a_sim/%s-%s_nouns_by_nums.csv'%(source,target), 'w')
    print "source = ", source
    print "target = ", target
    f.write("Source, Target, Method, Nouns, #pivots\n")
    for n in nums:
        print "#pivots = ", n
        nouns = runner(source,target,method,n) 
        f.write("%s, %s, %s, %f, %f\n"%(source,target,method,nouns,n))
        f.flush()
    f.close()
    pass

if __name__ == '__main__':
    source = "wsj"
    target = "answers"
    # method = "freq.NN"
    n = 500
    opt = 'r'
    # runner(source,target,method,n)
    # random_runner(source,target,method,n)
    methods = ['freq','mi','pmi','ppmi']
    # methods += ['un_freq','un_mi','un_pmi','un_ppmi']
    # batch_results_from_methods(source,target,methods,n)
    batch_dist_results_from_methods(source,target,methods,n)
    batch_results_from_NN_methods(source,target,methods,n)
    batch_results_from_f1_methods(source,target,methods,n,opt)