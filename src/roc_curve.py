# all the drawing methods
import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import pylab 
# import sklearn.metrics as metrics

# def compute_roc_and_auc(tpr,fpr):
#     # Compute ROC curve and ROC area for each class
#     roc_auc = []
#     n_classes = len(tpr)
#     for i in range(n_classes):
#         roc_auc[i] = metrics.auc(fpr[i], tpr[i])
#     print roc_auc
#     # Compute micro-average ROC curve and ROC area
#     # fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
#     # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#     pass

# draw roc
def draw_roc(tpr,fpr,auc):
    # roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    for i in range(len(tpr)):
        plt.plot(fpr[i], tpr[i], 'b', label = 'AUC = %0.2f' % auc[i])
    plt.legend(loc = 'upper right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('test.png')
    pass

# draw others vs distribution
def draw(x,y,y_label,source,target,pv_method,train_model,gamma):
    plt.figure(figsize=(11,5.5))
    index = np.arange(len(x))
    if train_model == 'combined':
        plt.title('%s-%s:%s,%s,$\gamma$=%s'%(source,target,convert(pv_method),train_model,digit_limit(gamma)))
    else:
        plt.title('%s-%s:%s,%s'%(source,target,convert(pv_method),train_model))
    plt.plot(index,y)
    pylab.xticks(index,x,rotation='vertical')
    plt.ylabel(y_label)
    plt.xlabel('POS_tags')
    plt.autoscale()
    plt.savefig('../work/a_sim/pic/%s-%s_%s_%s_%s.png'%(source,target,pv_method,train_model,y_label))
    pass

def draw_prf(x,ys,y_labels,source,target,pv_method,train_model,gamma):
    plt.figure(figsize=(11,5.5))
    index = np.arange(len(x))
    if train_model == 'combined':
        plt.title('%s-%s:%s,%s,$\gamma$=%s'%(source,target,convert(pv_method),train_model,digit_limit(gamma)))
    else:
        plt.title('%s-%s:%s,%s'%(source,target,convert(pv_method),train_model))
    i = 0
    for y in ys:
        plt.plot(index,y,label = y_labels[i])
        i+=1
    plt.legend(loc = 'lower right')
    pylab.xticks(index,x,rotation='vertical')
    plt.xlabel('POS_tags')
    plt.savefig('../work/a_sim/pic/prf/%s-%s_%s_%s.png'%(source,target,pv_method,train_model))
    pass

# draw f1 for pv_methods
def draw_methods(x,ys,y_labels,source,target,method,train_model,gamma):
    plt.figure(figsize=(9,5.5))
    index = np.arange(len(x))
    if 'x' in y_labels:
        if train_model == 'combined':
            plt.title('%s-%s:%s,%s,$\gamma$=%s'%(source,target,convert(method),train_model,digit_limit(gamma)),size=22)
        else:
            plt.title('%s-%s:%s,%s'%(source,target,convert(method),train_model),size=22)
        i = 0
        for y in ys:
            plt.plot(index,y,label = y_labels[i],linewidth=3.0)
            i+=1
        plt.legend(loc = 'upper right')
        pylab.xticks(index,x,rotation=45)
        plt.xlabel('POS_tags')
        plt.ylabel('F-score',size=22)
        plt.autoscale()
        plt.ylim([0,1.0])
        plt.savefig('../work/a_sim/pic/f1/%s-%s_%s_%s.png'%(source,target,method,train_model))
    else:
        if train_model == 'combined':
            plt.title('%s-%s:%s,%s,$\gamma$=%s'%(source,target,method,train_model,digit_limit(gamma)),size=22)
        else:
            plt.title('%s-%s:%s,%s'%(source,target,method,train_model),size=22)
        i = 0
        for y in ys:
            plt.plot(index,y,label = convert(y_labels[i]),linewidth=3.0)
            i+=1
        plt.legend(loc = 'upper right')
        pylab.xticks(index,x,rotation=45)
        plt.autoscale()
        plt.ylim([0,1.0])
        plt.xlabel('POS_tags')
        plt.ylabel('F-score',size=22)
        plt.savefig('../work/a_sim/pic/f1/%s-%s_%s_%s.png'%(source,target,method,train_model))
    pass

# convert names
def convert(method):
    if "landmark_" in method: 
        if method.replace("_pretrained","").replace("landmark_","") == "word2vec":
            return "T-CBOW"
        elif method.replace("_pretrained","").replace("landmark_","") == "glove":
            return "T-GloVe"
        else:
            return "T-Wiki"
    else:
        if "un_" in method:
            return "%s$_U$" % method.replace("un_","").upper()
        else:
            return "%s$_L$" % method.upper()

def digit_limit(tmp):
    return '%.1f'%tmp if (tmp>0.1 or tmp==0) else '$10^{%d}$'%(math.log10(tmp)-1)

def convert