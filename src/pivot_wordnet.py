from nltk.corpus import wordnet

# determine whether given word is noun (1) or not (0)
def is_noun(word):
    syns=wordnet.synsets(word)
    return 1 if str(syns[0].name()).split('.')[1]=='n' else 0

# read a list of words 
# to decide how many nouns (distribution) in the selected pivots 
def count_nouns(selected_pivots):
    return float(sum(1 for word in selected_pivots if is_noun(word)==1))/float(len(pivots))

def runner():
    pass

if __name__ == '__main__':
