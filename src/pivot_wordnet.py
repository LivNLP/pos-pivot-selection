from nltk.corpus import wordnet
import pos_data

# determine whether given word is noun (1) or not (0)
def is_noun(word):
    syns=wordnet.synsets(word)
    
    # if the word is in the datasets
    return 1 if str(syns[0].name()).split('.')[1]=='n' else 0

# read a list of words 
# to decide how many nouns (distribution) in the selected pivots 
def count_nouns(selected_pivots):
    return float(sum(1 for word in selected_pivots if is_noun(word)==1))/float(len(pivots))

def runner(source,target,method,n):
    features = pos_data.load_obj(source,target,method) if 'landmark' not in method else pos_data.load_obj(source,target,'/test/'+method)
    pivots = dict(features[:n]).keys()
    print count_nouns(pivots)
    pass

# different methods
def batch_results_from_methods(source,target,methods,n):
    print "source = ", source
    print "target = ", target
    for method in methods:
        print "method = ", method
        runner(source,target,method,n)
    pass

# different number of pivots
def batch_reuslts_from_numbers(source,target,method,nums):
    print "source = ", source
    print "target = ", target
    for n in nums:
        print "#pivots = ", n
        runner(source,target,method,n)    
    pass

if __name__ == '__main__':
    source = "wsj"
    target = "answers"
    method = "freq"
    n = 500
    runner(source,target,method,n)