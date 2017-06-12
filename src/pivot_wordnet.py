from nltk.corpus import wordnet
import pos_data
import random

# determine whether given word is noun (1) or not (0)
def is_noun(word):
    syns=wordnet.synsets(word)
    # if the word is in the datasets
    if syns:
        return 1 if str(syns[0].name()).split('.')[1]=='n' else 0
    else:
        return 0

# read a list of words 
# to decide how many nouns (distribution) in the selected pivots 
def count_nouns(selected_pivots):
    return float(sum(1 for word in selected_pivots if is_noun(word)==1))/float(len(selected_pivots))

def runner(source,target,method,n):
    features = pos_data.load_obj(source,target,method) if 'landmark' not in method else pos_data.load_obj(source,target,'/test/'+method)
    pivots = dict(features[:n]).keys() if n >0 else dict(features[n:]).keys()
    print count_nouns(pivots),n
    return count_nouns(pivots)

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
        f.write("%s, %S, %s, %f, %f\n"%(source,target,method,nouns,n))
        f.flush()
    f.close()
    pass

# different dist methods
def batch_dist_results_from_methods(source,target,methods,n):
    f = open('../work/dist_sim/%s-%sdist_nouns.csv'%(source,target), 'w')
    print "source = ", source
    print "target = ", target
    pv_method = "dist/"+method
    f.write("Source, Target, Method, Nouns, #pivots\n")
    for method in methods:
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
    # runner(source,target,method,n)
    # random_runner(source,target,method,n)
    methods = ['freq','mi','pmi','ppmi']
    batch_results_from_methods(source,target,methods,n)
    batch_dist_results_from_methods(source,target,methods,n)