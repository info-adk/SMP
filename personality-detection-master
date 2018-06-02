# coding: utf-8
# python version: 2.7
# date:2018.5.24

import csv
import re
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import numpy as np
from collections import OrderedDict

# ---------参数设置---------
csv_path = "/home/info/Info/personality-detection-master/essays.csv"
kernel = "linear"
isTenModel = False
vector_size = 200
# -------------------------

label_name = []
result1 = {} #accuracy
result2 = {} #recall
result3 = {} #f1
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9,!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip().lower()
kf = KFold(n_splits=10)

# 初始化存储结果的数据结构result
f = csv.reader(open(csv_path,"r"))
for line in f:
    label_name = line[2:]
    for i in range(len(label_name)):
        result1[label_name[i]] = []
        result2[label_name[i]] = []
        result3[label_name[i]] = []
    break


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9,!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip().lower()
def metrics_result(actual, predict):
    accuracy = round(metrics.precision_score(actual, predict,average='weighted'),3) #round() 第二个参数为精确小数点后几位
    recall = round(metrics.recall_score(actual, predict,average='weighted'),3)
    f1 = round(metrics.f1_score(actual, predict,average='weighted'),3)
    return accuracy,recall,f1

def read_data(csv_path):  #提供路经
    count = 0
    data,label_list= [],[]
    csv_file = csv.reader(open(csv_path,"r"))
    for line in csv_file:
        if csv_file.line_num == 1:
            label_name = line[2:]
            continue   # 跳过第一行
        authid = line[0]
        content = line[1]
        one_text_label = [line[2],line[3],line[4],line[5],line[6]] # 五个标签
        label_list.append(one_text_label)
        content = clean_str(content)
        data.append(TaggedDocument(content.split(),tags=[count])) # tags必须是可遍历的
        count += 1
    return data,label_list

# model.save("/home/iiip/Quincy/personality/personality-detection-master/doc2vec_model_200.model")
# model = Doc2Vec.load("/home/iiip/Quincy/personality/personality-detection-master/doc2vec_model_200.model")


def ten_fold_svm(model,kernel=kernel,label_name=[],label_list=[]): # 提供训练好的d2v模型
    for number in range(len(label_name)):  #针对每一个标签
        ave_accuracy = 0
        ave_recall = 0
        ave_f1 = 0

        for train_index,test_index in kf.split(data): # 10-fold
            print("-"),
            train,test = data_array[train_index].tolist(),data_array[test_index].tolist() #kf处理后是array,用tolist()

            train_X,train_y = [],[]
            test_X,test_y = [],[]

            for id in xrange(len(train)):  # xrange是生成器，减小内存消耗
                infer_vector = model.infer_vector(train[id][0])
                train_X.append(infer_vector)
                train_y.append(label_list[train[id][1][0]][number]) #label_list[train[id][1][0]][number] 第id个文档的第number个标签

            for id in xrange(len(test)):
                infer_vector = model.infer_vector(test[id][0])
                test_X.append(infer_vector)
                test_y.append(label_list[test[id][1][0]][number])


            clf = SVC(kernel=kernel)
            clf.fit(train_X,train_y)
            y_predict = clf.predict(test_X)

            accuracy,recall,f1 = metrics_result(test_y,y_predict)
            ave_accuracy+=accuracy
            ave_recall+=recall
            ave_f1+=f1

        print
        print label_name[number]
        print "ave_accuracy: ",ave_accuracy/10 #单次模型10折结果
        print "ave_recall: ",ave_recall/10
        print "ave_f1: ",ave_f1/10
        result1[label_name[number]].append(ave_accuracy/10)
        result2[label_name[number]].append(ave_recall/10)
        result3[label_name[number]].append(ave_f1/10)

def print_final_score(label_name):
    get_average = lambda x:1.0*sum(x)/len(x)
    print "----------------------------------------"
    print "10model average result"
    print
    for label in label_name:
        acc = get_average(result1[label])
        rec = get_average(result2[label])
        f1 = get_average(result3[label])

        print str(label)
        print "accuracy:",str(acc)
        print "recall:",str(rec)
        print "f1:",str(f1)



if __name__ == '__main__':
    data,label_list = read_data(csv_path)
    data_array = np.array(data)

    if isTenModel:
        for i in range(10): # 训练10次特征模型
            model = Doc2Vec(data, vector_size=vector_size, workers=8)
            model.train(data, total_examples=model.corpus_count, epochs=100)
            ten_fold_svm(model,kernel=kernel,label_name=label_name,label_list=label_list)
        print_final_score(label_name)
    else:
        model = Doc2Vec(data, vector_size=vector_size, workers=8)
        model.train(data, total_examples=model.corpus_count, epochs=100)
        ten_fold_svm(model, kernel=kernel, label_name=label_name, label_list=label_list)
