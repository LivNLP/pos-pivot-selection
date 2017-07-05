import numpy
import pos_data
# import classify_pos
import re
import sys, math, subprocess, time
from tabulate import tabulate
import roc_curve
from sklearn.metrics import classification_report
# from decimal import *

# return a list of all the labels from a output file or test file
def read_labels(fname):
    input_file = open(fname,'r')
    labels = [line.strip().split()[0] for line in input_file]
    return labels

# compare the labels between predicted from trained model and test data
def compare_labels(predict_labels,target_labels,old_tag_list,tag_dist):
    tag_list = set(predict_labels)|set(target_labels)
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
        inv_f1 = inverse_f1(f1)
        w = weight_score(f1)
        #acc = accuracy(tp,tn,fp,fn)
        # fpr = fallout(tn,fp)
        # auc = area_under_curve(r,inverse_recall(tn,fp))
        new_tag=number_to_tag(int(pos_tag),old_tag_list) 
        dist = dict(tag_dist).get(new_tag,0)
        result_list.append([new_tag,dist,p,r,inv_f1,f1,w])
    return result_list

def precision(tp,fp):
    return float(tp)/float(tp+fp) if tp+fp != 0 else 0

# tpr
def recall(tp,fn):
    return float(tp)/float(tp+fn) if tp+fn != 0 else 0

# tnr
def inverse_recall(tn,fp):
    return float(tn)/float(tn+fp) if tn+fp != 0 else 0

# fpr
def fallout(tn,fp):
    return float(fp)/float(tn+fp) if tn+fp != 0 else 0

def f1_score(precision,recall):
    return float(2.0*(precision*recall))/(float)(precision+recall) if precision+recall != 0 else 0

# r
def inverse_f1(f1_score):
    return float(1.0/f1_score) if f1_score != 0 else 0

#  1 / (1 + exp(-r)) - 0.5
def weight_score(f1_score):
    return float(1.0/(1.0+numpy.exp(-inverse_f1(f1_score))))-0.5 if 1.0+numpy.exp(-inverse_f1(f1_score))!=0 else 0

def accuracy(tp,tn,fp,fn):
    return float(tp+tn)/float(tp+tn+fp+fn)

def area_under_curve(recall,inverse_recall):
    # Recall = TPR, Inverse Recall = TNR 
    return float(recall+inverse_recall)/2.0

def sort_results(index,result_list):
    # return result_list.sort(lambda x,y: -1 if x[index] > y[index] else 1)
    return sorted(result_list,key=lambda x: x[index],reverse = True)

def create_table(table):
    # table = sort_results(index,result_list)
    headers = ["POS_tag","Distribution","Precision","Recall","Inverse_F1","F1","w"]
    # print result_list
    # add the avg as last line
    avg_list = []
    for i in range(1,len(headers)):
        tmp = [x[i] for x in table]
        # print numpy.mean(tmp)
        avg_list.append(numpy.mean(tmp))
    table.append(['[AVG]']+avg_list)
    
    tab = tabulate(table,headers,floatfmt=".4f")
    print tab
    return tab

"""
input: 
source_domain, target_domain, 
pivot_selection_method, train_model, 
order_by_index: 1-distribution,
gamma: mixing parameter for SCL pivot predictors.
"""
def evaluate_table(source,target,pv_method,train_model,index,gamma):
    print "source = ", source
    print "target = ", target
    print "pv_method: ", pv_method
    print "model: ", train_model
    print

    # test the trained model to generate output: predict_labels
    # combined
    model_file = '../work/%s/%s-%s/model.SCL.%f' % (pv_method,source,target,gamma)
    test_file = '../work/%s/%s-%s/testVects.SCL' % (pv_method,source,target)
    # implicit: word embeddings
    if train_model == "implicit":
        model_file = '../work/%s-%s/model.NA' % (source, target)
        test_file = '../work/%s-%s/testVects.NA' % (source, target)
    # explicit: SCL pivot predictors
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
    res_list = sort_results(index,compare_labels(predict_labels,target_labels,tag_list,tag_dist))
    tab = create_table(res_list)
    # draw_roc(res_list)
    # draw_prf(res_list[:len(tag_list)],source,target,pv_method,train_model,gamma)
    # for i in range(2,7):
    #     draw(res_list[:len(tag_list)],i,source,target,pv_method,train_model,gamma)
    # draw(res_list[:len(tag_list)],6,source,target,pv_method,train_model)
    # f = open("../work/a_sim/%s-%s_%s_table_%s"%(source,target,pv_method,train_model),"w")
    # pv_method = pv_method.replace("dist/","")
    # f = open("../work/dist_sim/%s-%s_%s_table_%s"%(source,target,pv_method,train_model),"w")
    # f.write(tab)
    # f.close()
    return res_list

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

# draw methods
def draw_roc(result_list):
    # get list of tpr and fpr from the result_list
    tpr = [x[3] for x in result_list]
    fpr = [x[4] for x in result_list]
    auc = [x[6] for x in result_list]
    roc_curve.draw_roc(tpr,fpr,auc)
    # roc_curve.compute_roc_and_auc(tpr,fpr)
    pass

# 1- dist, 2-p, 3-r(tpr), 4-fo(fpr), 5-f1, 6-auc
def draw(result_list,index,source,target,pv_method,train_model,gamma):
    # get distribution and values from the result_list
    # dist = [x[1] for x in result_list]
    tags = [x[0] for x in result_list]
    y_scores = [x[index] for x in result_list]
    y_labels = ["POS_tag","Distribution","Precision","Recall","Inverse_F1","F1","w"]
    y_label = y_labels[index]
    roc_curve.draw(tags,y_scores,y_label,source,target,pv_method,train_model,gamma)
    pass

def draw_prf(result_list,source,target,pv_method,train_model,gamma):
    tags = [x[0] for x in result_list]
    p = [x[2] for x in result_list]
    r = [x[3] for x in result_list]
    f1= [x[5] for x in result_list]
    ys=[p,r,f1]
    y = ["POS_tag","Distribution","Precision","Recall","Inverse_F1","F1","w"]
    y_labels=[y[2],y[3],y[5]]
    roc_curve.draw_prf(tags,ys,y_labels,source,target,pv_method,train_model,gamma)
    pass

def draw_f1_for_methods(source,target,pv_methods,method,train_model,gamma):
    ys = []
    tags = []
    new_methods = pv_methods
    if 'x' in pv_methods:
        # "f1/w/"+method, drop w(x)
        new_methods = [method,'dist/'+method,"f1/r/"+method, method+".NN"]
    else:
        if method == 'q(x)':
            new_methods = ['dist/'+pv for pv in pv_methods]
        elif method == 'r(x)':
            new_methods = ['f1/r/'+pv for pv in pv_methods]
        elif method == 'w(x)':
            new_methods = ['f1/w/'+pv for pv in pv_methods]
        elif method == 'x.NN':
            new_methods = [pv+'.NN' for pv in pv_methods]
    for pv_method in new_methods:
        res_list = evaluate_table(source,target,pv_method,train_model,1,gamma)
        f1 = [x[5] for x in res_list]
        tags = [x[0] for x in res_list]
        ys.append(f1)
    y_labels = pv_methods
    roc_curve.draw_methods(tags,ys,y_labels,source,target, method, train_model,gamma)
    pass

# print graphs
# e.g. FREQ-L: x, q(x), r(x), w(x), x.NN
def print_graphs_single_pv():
    source = 'wsj'
    target = 'answers'
    train_model = 'combined'
    gamma = 1
    # drop w(x) method on the graph,'w(x)'
    methods = ['x','q(x)','r(x)','x.NN']
    pv_methods=['freq','mi','pmi','ppmi']
    for pv_method in pv_methods:
        draw_f1_for_methods(source,target,methods,pv_method,train_model,gamma)
    pass

# e.g. x: FREQ-L, MI-L, PMI-L. PPMI-L
def print_graphs_single_method():
    source = 'wsj'
    target = 'answers'
    train_model = 'combined'
    gamma = 1
    # drop w(x) method on the graph,'w(x)'
    methods = ['x','q(x)','r(x)','x.NN']
    pv_methods=['freq','mi','pmi','ppmi']
    for method in methods:
        draw_f1_for_methods(source,target,pv_methods,method,train_model,gamma)
    pass

#unlabelled
def print_graph_unlabelled():
    source = 'wsj'
    target = 'answers'
    train_model = 'combined'
    gamma = 1
    method = 'un_x'
    pv_methods=['un_freq','un_mi','un_pmi','un_ppmi']
    draw_f1_for_methods(source,target,pv_methods,method,train_model,gamma)
    pass

# test methods
def test_sort():
    result_list = [['a',3,2,1],['b',1,2,2],['c',2,3,1]]
    print sort_results(1,result_list)
    pass

# batch f1 score for method implict,explicit and combined
def batch_f1_results(source,target,pv_method):
    f = open('../work/a_sim/%s-%s_F1.%s.csv'%(source,target,pv_method), 'w')
    print "Sum up results..."
    f.write("model, F1 score\n")
    train_models = ['explicit','implicit','combined']
    index = 1
    gamma = 1
    for train_model in train_models:
        res_list = evaluate_table(source,target,pv_method,train_model,index,gamma)
        tmp = [x[5] for x in res_list]
        avg_f1 = numpy.mean(tmp)
        print train_model,avg_f1
        f.write("%s, %f\n"%(train_model,avg_f1))
        f.flush()
    f.close()
    pass

# distribution
def batch_dist_f1_results(source,target,pv_method):
    f = open('../work/dist_sim/%s-%sdist_F1.%s.csv'%(source,target,pv_method), 'w')
    print "Sum up results..."
    f.write("model, F1 score\n")
    train_models = ['explicit','implicit','combined']
    index = 1
    gamma = 1
    pv_method = 'dist/'+pv_method
    for train_model in train_models:
        res_list = evaluate_table(source,target,pv_method,train_model,index,gamma)
        tmp = [x[5] for x in res_list]
        avg_f1 = numpy.mean(tmp)
        print train_model,avg_f1
        f.write("%s, %f\n"%(train_model,avg_f1))
        f.flush()
    f.close()
    pass

# F-score
def batch_f1_results_with_opt(source,target,pv_method,opt):
    f = open('../work/f1_sim/%s/%s-%sf1_F1.%s.csv'%(opt,source,target,pv_method), 'w')
    print "Sum up results..."
    f.write("model, F1 score\n")
    train_models = ['explicit','implicit','combined']
    index = 1
    gamma = 1
    pv_method = ("f1/%s/"%opt)+pv_method
    for train_model in train_models:
        res_list = evaluate_table(source,target,pv_method,train_model,index,gamma)
        tmp = [x[5] for x in res_list]
        avg_f1 = numpy.mean(tmp)
        print train_model,avg_f1
        f.write("%s, %f\n"%(train_model,avg_f1))
        f.flush()
    f.close()
    pass

# gamma results for unbalanced function
def batch_gamma_results(source,target,pv_method):
    f = open('../work/a_sim/%s-%sgamma_F1.%s.csv'%(source,target,pv_method), 'w')
    print "Generating results for different gamma values..."
    f.write("gamma, F1 score\n")
    gammas = [0.01,0.1,1,10,100]
    for gamma in gammas:
        model_file = '../work/%s/%s-%s/model.SCL.%f' % (pv_method,source,target,gamma)
        test_file = '../work/%s/%s-%s/testVects.SCL' % (pv_method,source,target)
        testLBFGS(test_file,model_file)
        output = '../work/output_eval'
        predict_labels = read_labels(output)
        target_labels = read_labels(test_file)
        tag_list = generate_tag_list(source,target)
        # print tag_list
        tag_dist = pos_data.compute_dist(source)
        # default sort by distribution
        res_list = sort_results(1,compare_labels(predict_labels,target_labels,tag_list,tag_dist))
        tmp = [x[5] for x in res_list]
        avg_f1 = numpy.mean(tmp)
        print gamma,avg_f1
        f.write("%f, %f\n"%(gamma,avg_f1))
        f.flush()
    f.close()
    pass

# gamma results for dist (balanced) function
def batch_dist_gamma_results(source,target,method):
    pv_method = "dist/"+method
    f = open('../work/dist_sim/%s-%sdistgamma_F1.%s.csv'%(source,target,method), 'w')
    print "Generating results for different gamma values..."
    f.write("gamma,F1 score\n")
    gammas = [0.01,0.1,1,10,100]
    for gamma in gammas:
        model_file = '../work/%s/%s-%s/model.SCL.%f' % (pv_method,source,target,gamma)
        test_file = '../work/%s/%s-%s/testVects.SCL' % (pv_method,source,target)
        testLBFGS(test_file,model_file)
        output = '../work/output_eval'
        predict_labels = read_labels(output)
        target_labels = read_labels(test_file)
        tag_list = generate_tag_list(source,target)
        # print tag_list
        tag_dist = pos_data.compute_dist(source)
        # default sort by distribution
        res_list = sort_results(1,compare_labels(predict_labels,target_labels,tag_list,tag_dist))
        tmp = [x[5] for x in res_list]
        avg_f1 = numpy.mean(tmp)
        print gamma,avg_f1
        f.write("%f, %f\n"%(gamma,avg_f1))
        f.flush()
    f.close()
    pass

def batch_f1_gamma_results(source,target,method,opt):
    pv_method = ("f1/%s/"%opt)+method
    f = open('../work/f1_sim/%s/%s-%sf1gamma_F1.%s.csv'%(opt,source,target,method), 'w')
    print "Generating results for different gamma values..."
    f.write("gamma,F1 score\n")
    gammas = [0.01,0.1,1,10,100]
    for gamma in gammas:
        model_file = '../work/%s/%s-%s/model.SCL.%f' % (pv_method,source,target,gamma)
        test_file = '../work/%s/%s-%s/testVects.SCL' % (pv_method,source,target)
        testLBFGS(test_file,model_file)
        output = '../work/output_eval'
        predict_labels = read_labels(output)
        target_labels = read_labels(test_file)
        tag_list = generate_tag_list(source,target)
        # print tag_list
        tag_dist = pos_data.compute_dist(source)
        # default sort by distribution
        res_list = sort_results(1,compare_labels(predict_labels,target_labels,tag_list,tag_dist))
        tmp = [x[5] for x in res_list]
        avg_f1 = numpy.mean(tmp)
        print gamma,avg_f1
        f.write("%f, %f\n"%(gamma,avg_f1))
        f.flush()
    f.close()
    pass

def clas_rpt():
    source = "wsj"
    target = "answers"
    pv_method = "mi"
    model_file = '../work/%s/%s-%s/model_lexical.SCL' % (pv_method,source,target)
    test_file = '../work/%s/%s-%s/testVects_lexical.SCL' % (pv_method,source,target)
    testLBFGS(test_file,model_file)
    output = '../work/output_eval'
    predict_labels = read_labels(output)
    target_labels = read_labels(test_file)
    print(classification_report(target_labels, predict_labels))
    pass

def print_results():
    source = 'wsj'
    target = 'answers'
    # pv_method = 'freq.NN'
    pv_method = 'un_mi'
    # pv_method = 'mi'
    # pv_method = 'mi.NN'
    # train_model = 'implicit'
    train_models = ['explicit','implicit','combined']
    # train_model = 'explicit'
    # train_model = 'combined'
    index = 1
    gamma = 1
    for train_model in train_models:
        evaluate_table(source,target,pv_method,train_model,index,gamma)
    pass

def print_f1_results():
    source = 'wsj'
    target = 'answers'
    methods = ['freq','mi','pmi','ppmi']
    # methods = ["ppmi"]
    opt = 'r'
    # methods += ['un_freq','un_mi','un_pmi','un_ppmi']
    # pos_tag = 'NN'
    for pv_method in methods:
        # pv_method = '%s.%s'%(pv_method,pos_tag)
        # print "method = ", pv_method
        # batch_f1_results(source,target,pv_method)
        # batch_dist_f1_results(source,target,pv_method)
        batch_f1_results_with_opt(source,target,pv_method,opt)
    pass

def print_gamma_results():
    source = "wsj"
    target = "answers"
    methods = ['freq','mi','pmi','ppmi']
    opt = 'r'
    # methods = ['un_freq','un_mi','un_pmi','un_ppmi']
    # methods = ["ppmi"]
    # pv_method = "dist/mi"
    # pv_method = "un_mi"
    # pos_tag = 'NN'
    for pv_method in methods:
        # batch_dist_gamma_results(source,target,pv_method)
        # pv_method = '%s.%s'%(pv_method,pos_tag)
        # batch_gamma_results(source,target,pv_method)
        batch_f1_gamma_results(source,target,pv_method,opt)
    pass

if __name__ == '__main__':
    # print_results()
    # print_f1_results()
    # print_gamma_results()
    # test_sort()
    # clas_rpt()
    print_graphs_single_pv()
    print_graphs_single_method()
    # print_graph_unlabelled()
