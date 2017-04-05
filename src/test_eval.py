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
    for pos_tag in tag_list:
        
    return tp,tn,fp,fn

def precision(tp,fp):
    return tp/(float)(tp+fp) if tp+fp != 0 else 0

def recall(tp,fn):
    return tp/(float)(tp+fn) if tp+fn != 0 else 0

def f_score(precision,recall):
    return 2*(precision*recall)/(float)(precision+recall) if precision+recall != 0 else 0

