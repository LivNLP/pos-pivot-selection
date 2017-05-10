import numpy
import pos_data
# import classify_pos
import re
import sys, math, subprocess, time
from tabulate import tabulate
# from decimal import *

# return a list of all the labels from a output file or test file
def read_labels(fname):
    input_file = open(fname,'r')
    labels = [line.strip().split()[0] for line in input_file]
    return labels

# compare the labels between predicted from trained model and test data
def compare_labels(predict_labels,target_labels,old_tag_list,tag_dist):
    tag_list = set(predict_labels)&set(target_labels)
    result_list = []
    for pos_tag in tag_list:
        tp=0
        tn=0
        fp=0
        fn=0
        for i,predict_label in enumerate(predict_labels):
            target_label = target_labels[i]
            # true positive
            if predict_label == pos_tag and target_label == pos_tag:
                tp+=1
            # true negative
            elif predict_label != pos_tag and target_label != pos_tag:
                tn+=1
            # false positive
            elif predict_label == pos_tag and target_label != pos_tag:
                fp+=1
            # false negative
            elif predict_label != pos_tag and target_label == pos_tag:
                fn+=1
        p = precision(tp,fp)
        r = recall(tp,fn)
        f1 = f1_score(p,r)
        acc = accuracy(tp,tn,fp,fn)
        new_tag=number_to_tag(int(pos_tag),old_tag_list) 
        dist = dict(tag_dist).get(new_tag,0)
        result_list.append([new_tag,dist,p,r,f1,acc])
    return result_list

def precision(tp,fp):
    return float(tp)/(float)(tp+fp) if tp+fp != 0 else 0

def recall(tp,fn):
    return float(tp)/(float)(tp+fn) if tp+fn != 0 else 0

def f1_score(precision,recall):
    return float(2*(precision*recall))/(float)(precision+recall) if precision+recall != 0 else 0

def accuracy(tp,tn,fp,fn):
    return float(tp+tn)/float(tp+tn+fp+fn)

def sort_results(index,result_list):
    # return result_list.sort(lambda x,y: -1 if x[index] > y[index] else 1)
    return sorted(result_list,key=lambda x: x[index],reverse = True)

def create_table(index,result_list):
    table = sort_results(index,result_list)
    headers = ["POS_tag","Distribution","Precision","Recall","F1 Score","Accuracy"]
    # print result_list
    # add the avg as last line
    avg_list = []
    for i in range(1,len(headers)):
        tmp = [x[i] for x in result_list]
        # print numpy.mean(tmp)
        avg_list.append(numpy.mean(tmp))
    table.append(['AVG']+avg_list)
    
    tab = tabulate(table,headers,floatfmt=".4f")
    print tab
    return tab

def evaluate_table(source,target,pv_method,train_model,index):
    print "source = ", source
    print "target = ", target
    print "pv_method: ", pv_method
    print "model: ", train_model

    # test the trained model to generate output: predict_labels
    model_file = '../work/%s/%s-%s/model.SCL' % (pv_method,source,target)
    test_file = '../work/%s/%s-%s/testVects.SCL' % (pv_method,source,target)
    if train_model == "implicit":
        model_file = '../work/%s-%s/model.NA' % (source, target)
        test_file = '../work/%s-%s/testVects.NA' % (source, target)
    if train_model == "explicit":
        model_file = '../work/%s/%s-%s/model_lexical.SCL' % (pv_method,source,target)
        test_file = '../work/%s/%s-%s/testVects_lexical.SCL' % (pv_method,source,target)
    testLBFGS(test_file,model_file)
    output = '../work/output_eval'
    predict_labels = read_labels(output)
    target_labels = read_labels(test_file)
    tag_list = generate_tag_list(source,target)
    # print tag_list
    tag_dist = pos_data.compute_dist(source)
    tab = create_table(index,compare_labels(predict_labels,target_labels,tag_list,tag_dist))
    f = open("../work/a_sim/%s-%s_%s_table_%s"%(source,target,pv_method,train_model),"w")
    f.write(tab)
    f.close()
    pass

def testLBFGS(test_file, model_file):
    output = '../work/output_eval'
    retcode = subprocess.check_output('~/liblinear-multicore-2.11-1/predict %s %s %s' %\
        (test_file,model_file,output), shell=True)
    return retcode


def number_to_tag(number,tag_list):
    return tag_list[number-1] if number-1 <= len(tag_list) else -1

def generate_tag_list(source,target):
    train_sentences = pos_data.load_preprocess_obj('%s-labeled'%source)
    test_sentences = pos_data.load_preprocess_obj('%s-test'%target)
    tag_list = list(set(pos_data.tag_list(train_sentences))&set(pos_data.tag_list(test_sentences)))
    # pos_data.save_obj(source,target,tag_list,"tag_list")
    return tag_list

# test methods
def test():
    result_list = [['a',3,2,1],['b',1,2,2],['c',2,3,1]]
    print sort_results(1,result_list)
    pass

if __name__ == '__main__':
    source = 'wsj'
    target = 'answers'
    # pv_method = 'freq'
    pv_method = 'un_freq'
    # train_model = 'implicit'
    train_models = ['implicit','explicit','combined']
    # train_model = 'explicit'
    # train_model = 'combined'
    index = 1
    for train_model in train_models:
        evaluate_table(source,target,pv_method,train_model,index)
    # test()