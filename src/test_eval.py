import numpy
import pos_data
import classify_pos
import re
import sys, math, subprocess, time

# return a list of all the labels from a output file or test file
def read_labels(source,target,method,fname):
    input_file = open(fname,'r')
    labels = [line.strip().split()[0] for line in input_file]
    return labels

def compare_labels(predict_labels,target_labels):
    tag_list = set(predict_labels)&set(target_labels)
    result_list = []
    for pos_tag in tag_list:
        tp,tn,fp,fn = 0
        for i,predict_label in enumerate(predict_labels):
            target_label = target_labels[i]
            # true positive
            if predict_label == pos_tag and target_label == pos_tag:
                tp++
            # true negative
            if predict_label != pos_tag and target_label != pos_tag:
                tn++
            # false positive
            if predict_label == pos_tag and target_label != pos_tag:
                fp++
            # false negative
            if predict_label != pos_tag and target_label == pos_tag:
                fn++
        p = precision(tp,fp)
        r = recall(tp,fn)
        f1 = f1_score(p,r)
        result_list.append([pos_tag,p,r,f1])
    return result_list

def precision(tp,fp):
    return tp/(float)(tp+fp) if tp+fp != 0 else 0

def recall(tp,fn):
    return tp/(float)(tp+fn) if tp+fn != 0 else 0

def f1_score(precision,recall):
    return 2*(precision*recall)/(float)(precision+recall) if precision+recall != 0 else 0

