#coding: utf-8
# python version: 2.7
# date:2018.5.24

import csv
import re
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import numpy as np

f = open("/home/info/Info/personality-detection-master/essays.csv","r")
csv_file = csv.reader(f)

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

data,train,test = [],[],[]  # 放特征向量
label_name = []
label_list=[] #存放每个用户对应标签
count = 0

def metrics_result(actual, predict):
    accuracy = round(metrics.precision_score(actual, predict,average='weighted'),3) #round() 第二个参数为精确小数点后几位
    recall = round(metrics.recall_score(actual, predict,average='weighted'),3)
    f1 = round(metrics.f1_score(actual, predict,average='weighted'),3)
    return accuracy,recall,f1

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

# model.save("/home/iiip/Quincy/personality/personality-detection-master/doc2vec_model_200.model")
# model = Doc2Vec.load("/home/iiip/Quincy/personality/personality-detection-master/doc2vec_model_200.model")


kf = KFold(n_splits=10)
data_array = np.array(data)
result1 = {"cEXT":[],"cNEU":[],"cAGR":[],"cCON":[],"cOPN":[]} #accuracy
result2 = {"cEXT":[],"cNEU":[],"cAGR":[],"cCON":[],"cOPN":[]} #recall
result3 = {"cEXT":[],"cNEU":[],"cAGR":[],"cCON":[],"cOPN":[]} #f1

for i in range(10): # 训练10次特征模型
    model = Doc2Vec(data, vector_size=200, workers=8)
    model.train(data, total_examples=model.corpus_count, epochs=100)
    
    for number in range(len(label_name)):  #针对每一个标签
        ave_accuracy = 0
        ave_recall = 0
        ave_f1 = 0

        for train_index,test_index in kf.split(data): # 10-fold
            print("-"),
            train,test = data_array[train_index].tolist(),data_array[test_index].tolist() #kf处理后是array,用tolist()

            train_X,train_y1 = [],[]
            test_X,test_y1 = [],[]

            for id in xrange(len(train)):  # xrange是生成器，减小内存消耗
                infer_vector = model.infer_vector(train[id][0])
                train_X.append(infer_vector)
                train_y1.append(label_list[train[id][1][0]][number]) #label_list[train[id][1][0]][number] 第id个文档的第number个标签

            for id in xrange(len(test)):
                infer_vector = model.infer_vector(test[id][0])
                test_X.append(infer_vector)
                test_y1.append(label_list[test[id][1][0]][number])


            clf = SVC(kernel="linear")
            clf.fit(train_X,train_y1)
            y_predict = clf.predict(test_X)

            accuracy,recall,f1 = metrics_result(test_y1,y_predict)
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

with open("/home/info/Info/personality-detection-master/d2v_10fold_10model_essay.txt", "w") as f:
    get_average = lambda x:1.0*sum(x)/len(x)
    write_content = ""
    for label in result1:
        acc = get_average(result1[label])
        rec = get_average(result2[label])
        f1 = get_average(result3[label])
        write_content +=str(label)+"\n"+"accuracy: "+str(acc)+"\n"+"recall: "+str(rec)+"\n"+"f1: "+str(f1)+"\n"+"\n"
    f.write(write_content)
