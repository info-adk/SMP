# coding:utf-8

import csv
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np

f = open("/home/info/Info/personality-detection-master/PersonalityRecognizer/output/200g.csv","r") #向量
l = open("/home/info/Info/personality-detection-master/newdata/200g.csv","r") 

label_name = ["C","D","E","F","G"]
result = {}
for name in label_name:
    result[name] = 0
data = []
label = []
f_reader = csv.reader(f)
l_reader = csv.reader(l)


def metrics_result(actual, predict):
    accuracy = round(metrics.accuracy_score(actual, predict),3) #round() 第二个参数为精确小数点后几位
    return accuracy

for line in f_reader:
    line = line[1:]
    X = []
    for value in line:
        value = float(value)
        X.append(value)
    data.append(X)

for line in l_reader:
    if l_reader.line_num == 1:
        continue
    else:
        one_text_label = line[2:]
        label.append(one_text_label)

kf = KFold(n_splits=10,shuffle=True)
data = shuffle(data)
for num in range(5):
    key = label_name[num]
    for train_index,test_index in kf.split(data):
        data = np.array(data)
        label = np.array(label)
        train_X = data[train_index].tolist()
        train_y = [i[num] for i in label[train_index].tolist()]
        test_X = data[test_index].tolist()
        test_y = [i[num] for i in label[test_index].tolist()]

        clf = SVC(kernel="rbf")
        clf.fit(train_X,train_y)
        y_predict = clf.predict(test_X)

        score = metrics_result(y_predict,test_y)

        result[key] += 1.0*score/10

    print key+":",result[key]
f.close()
l.close()
