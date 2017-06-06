"""
Only apply to pivot selection methods using labelled datasets:
FREQ_L, MI_L, PMI_L, PPMI_L

Difference between f1: 
q(x), q is not distribution, it's a normalized f1 weight
"""
import pos_data
import os
import pickle
import test_eval

# add f1 when sum up the scores
def sum_up_f1_labeled_scores(source,target,opt):
    src_labeled = pos_data.load_preprocess_obj('%s-labeled'%source)
    # tags = pos_data.tag_list(src_labeled)
    # all the labeled methods
    methods = ['freq','mi','pmi','ppmi']
    for method in methods:
        res_list=test_eval.evaluate_table(source,target,method,'combined',1,1)
        tags = [x[0] for x in res_list]
        print tags
        f1s = [x[4] for x in res_list] if opt=='r' else [x[6] for x in res_list]
        print f1s

        # loop tags to divide presets into groups
        freq_dict={}
        mi_dict={}
        pmi_dict={}
        ppmi_dict={}
        for idx,pos_tag in enumerate(tags):
            print "TAG = %s"% pos_tag
            f1 = f1s[idx]
            print "f1 = %f" % f1
            if method == 'freq':
                # print "FREQ-L"
                tmp = pos_data.select_pivots_freq_labeled_tag(source,target,pos_tag)
                freq_dict = pos_data.combine_dicts(freq_dict,multiply_f1(tmp,f1))
            elif method == 'mi':
                # print "MI-L"
                tmp = pos_data.select_pivots_mi_labeled_tag(source,target,pos_tag)
                mi_dict = pos_data.combine_dicts(mi_dict,multiply_f1(tmp,f1))           
            elif method == 'pmi':
                # print "PMI-L"
                tmp = pos_data.select_pivots_pmi_labeled_tag(source,target,pos_tag)
                pmi_dict = pos_data.combine_dicts(pmi_dict,multiply_f1(tmp,f1))           
            else:
                # print "PPMI-L"
                tmp = pos_data.select_pivots_ppmi_labeled_tag(source,target,pos_tag)
                ppmi_dict = pos_data.combine_dicts(ppmi_dict,multiply_f1(tmp,f1))
    freq_list = freq_dict.items()
    mi_list = mi_dict.items()
    pmi_list = pmi_dict.items()
    ppmi_list = ppmi_dict.items()

    freq_list.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    mi_list.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    pmi_list.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    ppmi_list.sort(lambda x, y: -1 if x[1] > y[1] else 1)

    print "FREQ", freq_list[:10]
    print "MI", mi_list[:10]
    print "PMI", pmi_list[:10]
    print "PPMI", ppmi_list[:10]
    save_f1_obj(source,target,freq_list,"%s/freq"%opt)
    save_f1_obj(source,target,mi_list,"%s/mi"%opt)
    save_f1_obj(source,target,pmi_list,"%s/pmi"%opt)
    save_f1_obj(source,target,ppmi_list,"%s/ppmi"%opt)
    pass

def multiply_f1(L,f1):
    # print L
    return dict([(x,f1*v) for (x,v) in L.iteritems()])

def compute_f1(opt):
    
    pass

# save and load f1 obj
def save_f1_obj(source,target,obj,name):
    filename = '../work/%s-%s/f1/%s.pkl'%(source,target,name)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print '%s saved'%filename

def load_f1_obj(source,target,name):
    with open('../work/%s-%s/f1/%s.pkl'%(source,target,name), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    source = 'wsj'
    target = 'answers'
    sum_up_f1_labeled_scores(source,target,'r')
    # sum_up_f1_labeled_scores(source,target,'w')