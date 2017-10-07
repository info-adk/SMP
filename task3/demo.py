# coding=utf-8

## 加载数据
from sklearn.datasets import load_svmlight_file
filename = "data/trainingset/valueMostAttributes.txt"
data = load_svmlight_file(filename)
X, y = data[0], data[1]
X = X.toarray()

## 划分数据集
from sklearn.model_selection import ShuffleSplit
rs = ShuffleSplit(n_splits=1, train_size=0.9, test_size=0.1, random_state=0)
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

## 构建分类器模型(0.003)
from sklearn.ensemble import RandomForestClassifier
classify_model = RandomForestClassifier(n_estimators=50, random_state=1)
y_trainset_label = []
for i in range (0, y_trainset.__len__(), 1):
    if y_trainset[i] < 0.003:
        y_trainset_label.append(0.0)
    else :
        y_trainset_label.append(1.0)
classify_model.fit(X_trainset, y_trainset_label)

classify_result = classify_model.predict(X_testset)

class_one = 0
class_two = 0
predict_one = 0
predict_two = 0

for i in range(0, classify_result.__len__(), 1):
    if y_testset[i] < 0.003:
        class_one += 1
        if classify_result[i] == 0.:
            predict_one += 1
    if y_testset[i] >= 0.003:
        class_two += 1
        if classify_result[i] == 1.:
            predict_two += 1
    # if result_001[i] != y_testset_001[i]:
    #     pro =classify_model_001.predict_proba(X_testset_001[i])
    #     print pro[0]
print class_one, predict_one
print class_two, predict_two

## 构建回归模型
### 构建0.003的回归模型
from sklearn.linear_model import BayesianRidge, RANSACRegressor, RidgeCV, Ridge, LassoLarsCV
X_trainset_0003 = []
y_trainset_0003 = []
for i in range(0, y_trainset.__len__(), 1):
    if y_trainset[i] < 0.003:
        X_trainset_0003.append(X_trainset[i])
        y_trainset_0003.append(y_trainset[i])

reg_0003 = LassoLarsCV()
reg_0003.fit(X_trainset_0003, y_trainset_0003)

X_testset_0003 = []
y_testset_0003 = []
for i in range(0, y_testset.__len__(), 1):
    if y_testset[i] < 0.003:
        X_testset_0003.append(X_testset[i])
        y_testset_0003.append(y_testset[i])
reg_0003_result = reg_0003.predict(X_testset_0003)
mse_0003 = 0.0
for i in range(0, y_testset_0003.__len__(), 1):
    print reg_0003_result[i], y_testset_0003[i]
    mse_0003 += abs(reg_0003_result[i] - y_testset_0003[i])
print mse_0003 / y_testset_0003.__len__()

### 构建大于0.003的回归模型
#### 构建0.05的分类模型

from sklearn.linear_model import BayesianRidge, RANSACRegressor, RidgeCV, Ridge, LassoLarsCV
X_trainset_001 = []
y_trainset_001 = []
for i in range(0, y_trainset.__len__(), 1):
    if y_trainset[i] >= 0.003 and y_trainset[i] < 0.01:
        X_trainset_001.append(X_trainset[i])
        y_trainset_001.append(0.0)
    if y_trainset[i] >= 0.01:
        X_trainset_001.append(X_trainset[i])
        y_trainset_001.append(1.0)
classify_model_001 = RandomForestClassifier(n_estimators=50, random_state=1)
classify_model_001.fit(X_trainset_001,y_trainset_001)

X_testset_001 = []
y_testset_001 = []
for i in range(0, y_testset.__len__(), 1):
    if y_testset[i] >= 0.003 and y_testset[i] < 0.01:
        X_testset_001.append(X_testset[i])
        y_testset_001.append(0.0)
    if y_testset[i] >= 0.01:
        X_testset_001.append(X_testset[i])
        y_testset_001.append(1.0)

above_001_num = 0
below_001_num = 0
predict_above_001 = 0
predict_below_001 = 0
num = 0
result_001 = classify_model_001.predict(X_testset_001)
for i in range(0, result_001.__len__(), 1):
    if y_testset_001[i] == 0.:
        below_001_num += 1
        if result_001[i] == 0.:
            predict_below_001 += 1
    if y_testset_001[i] == 1.:
        above_001_num += 1
        if result_001[i] == 1.0:
            predict_above_001 += 1
    if result_001[i] != y_testset_001[i]:
        pro =classify_model_001.predict_proba(X_testset_001[i])
        print pro[0]

print below_001_num, predict_below_001
print above_001_num, predict_above_001
print num
#
### 构建大于0.1的回归模型
from sklearn.linear_model import BayesianRidge, RANSACRegressor, RidgeCV, Ridge, LassoLarsCV
X_trainset_1 = []
y_trainset_1 = []
for i in range(0, y_trainset.__len__(), 1):
    if y_trainset[i] >= 0.01 and y_trainset[i] < 1:
        X_trainset_1.append(X_trainset[i])
        y_trainset_1.append(y_trainset[i])

reg_1 = LassoLarsCV(max_n_alphas=10,positive=True)

reg_1.fit(X_trainset_1, y_trainset_1)

X_testset_1 = []
y_testset_1 = []
for i in range(0, y_testset.__len__(), 1):
    if y_testset[i] >= 0.01 and y_testset[i] < 1:
        X_testset_1.append(X_testset[i])
        y_testset_1.append(y_testset[i])
reg_1_result = reg_1.predict(X_testset_1)
mse_1 = 0.0
for i in range(0, y_testset_1.__len__(), 1):
    print reg_1_result[i], y_testset_1[i]
    mse_1 += abs(reg_1_result[i] - y_testset_1[i])
print mse_1 / y_testset_1.__len__()

