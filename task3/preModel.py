# coding=utf-8

## 加载数据
from sklearn.datasets import load_svmlight_file
filename = "data/trainingset/valueMostAttributes.txt"
data = load_svmlight_file(filename)
X, y = data[0], data[1]
X = X.toarray()

## 划分数据集
from sklearn.model_selection import ShuffleSplit
rs = ShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=1)
rs.get_n_splits(X)
X_trainset = None
X_testset = None
y_trainset = None
y_testset = None

for train_index, test_index in rs.split(X, y):
    X_trainset, X_testset = X[train_index], X[test_index]
    y_trainset, y_testset = y[train_index], y[test_index]

## 对训练集归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False).fit(X_trainset)
X_trainset = scaler.transform(X_trainset)

## 构建分类器模型(0.003)
from sklearn.ensemble import RandomForestClassifier
classify_model_0003 = RandomForestClassifier(n_estimators=50, random_state=1)
y_trainset_label = []
num_11 = 0
num_12 = 0
for i in range (0, y_trainset.__len__(), 1):
    if y_trainset[i] < 0.003:
        y_trainset_label.append(0.0)
        num_11 += 1
    else :
        y_trainset_label.append(1.0)
        num_12 += 1
print num_11, num_12
classify_model_0003.fit(X_trainset, y_trainset_label)

## 构建分类器模型(0.01)
X_trainset_001 = []
y_trainset_001 = []
num_1 = 0
num_2 = 0
for i in range(0, y_trainset.__len__(), 1):
    if y_trainset[i] >= 0.003 and y_trainset[i] < 0.015:
        X_trainset_001.append(X_trainset[i])
        y_trainset_001.append(0.0)
        num_1 += 1
    if y_trainset[i] >= 0.015:
        X_trainset_001.append(X_trainset[i])
        y_trainset_001.append(1.0)
        num_2 += 1
print num_1,num_2
classify_model_001 = RandomForestClassifier(n_estimators=55, random_state=1)
classify_model_001.fit(X_trainset_001,y_trainset_001)

### 构建0.003的回归模型
from sklearn.linear_model import LassoLarsCV,BayesianRidge
X_trainset_0003 = []
y_trainset_0003 = []
for i in range(0, y_trainset.__len__(), 1):
    if y_trainset[i] < 0.003:
        X_trainset_0003.append(X_trainset[i])
        y_trainset_0003.append(y_trainset[i])
reg_0003 = LassoLarsCV(max_n_alphas=100,positive=True)
reg_0003.fit(X_trainset_0003, y_trainset_0003)

### 构建0.003-0.01的回归模型
from sklearn.linear_model import LassoLarsCV
X_trainset_001 = []
y_trainset_001 = []
for i in range(0, y_trainset.__len__(), 1):
    if y_trainset[i] >= 0.003 and y_trainset[i] < 0.015:
        X_trainset_001.append(X_trainset[i])
        y_trainset_001.append(y_trainset[i])
reg_001 = LassoLarsCV(max_n_alphas=100,cv=10)
reg_001.fit(X_trainset_001, y_trainset_001)


### 构建大于0.01的回归模型
from sklearn.linear_model import BayesianRidge, RANSACRegressor, RidgeCV, Ridge, LassoLarsCV
X_trainset_1 = []
y_trainset_1 = []
for i in range(0, y_trainset.__len__(), 1):
    if y_trainset[i] >= 0.015 and y_trainset[i] < 1:
        X_trainset_1.append(X_trainset[i])
        y_trainset_1.append(y_trainset[i])
reg_1 = LassoLarsCV(max_n_alphas=100,positive=True)
reg_1.fit(X_trainset_1, y_trainset_1)


## 预测
mse = 0.0
lines = ''

from sklearn.datasets import load_svmlight_file
filename = "data/trainingset/oneThousandProperties.txt"
data = load_svmlight_file(filename)
X_testset, y_testset = data[0], data[1]
X_testset = X_testset.toarray()

for i in range(0, y_testset.__len__(), 1):
    predict_x = 0.0
    test_x = X_testset[i]
    test_x = scaler.transform(test_x)
    one_classify_pro = classify_model_0003.predict_proba(test_x)
    probe = one_classify_pro[0]
    if probe[0] - probe[1] > 0.4:
        predict_x = reg_0003.predict(test_x)
    elif probe[1] - probe[0] > 0.4:
        two_classify_pro = classify_model_001.predict_proba(test_x)
        probe_two = two_classify_pro[0]
        if probe_two[0] - probe_two[1] > 1:
            predict_x = reg_001.predict(test_x)
        elif probe_two[1] - probe_two[0] > 1:
            predict_x = reg_1.predict(test_x)
        else:
            if probe_two[1] > probe_two[0]:
                predict_x = 0.000 * probe_two[0] * reg_001.predict(test_x) + probe_two[1] * reg_1.predict(test_x)
            else:
                predict_x = probe_two[0] * reg_001.predict(test_x) + 0.45 * probe_two[1] * reg_1.predict(test_x)

    else:
        if probe[0] > probe[1]:
            predict_x = probe[0]* reg_0003.predict(test_x) + 0.65 *probe[1] * reg_001.predict(test_x)
        else:
            predict_x = 1.825 * probe[0] * reg_0003.predict(test_x) + probe[1] * reg_001.predict(test_x)
    print predict_x , y_testset[i]
    mse += abs(predict_x - y_testset[i])
    # lines += str(round(predict_x,4)) + ',' + str(y_testset[i])+ '\n'
    lines += str(round(predict_x,4)) + '\n'
print lines
print mse / y_testset.__len__()

f = open('data/valid/growresult.txt','w')
f.write(lines)
f.close()


