# coding=utf-8


## 保存模型
from sklearn.externals import joblib

## 读取svmfile文件
from sklearn.datasets import load_svmlight_file
filename = "./data/trainingset/fourclass.txt"
data = load_svmlight_file(filename)
X, y = data[0], data[1] # X为特征属性, y为目标属性

## 归一化处理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False).fit(X)
joblib.dump(scaler, "classify_scaler_0003.m")


## 特征提取
# from sklearn.feature_selection import chi2, SelectKBest
# k_best = SelectKBest(chi2, k=400)
# X = k_best.
# print SelectKBest(chi2, k=400)

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

## 训练模型
from sklearn.ensemble import RandomForestClassifier
classify_model = RandomForestClassifier(n_estimators=50, random_state=1)
classify_model.fit(X_trainset, y_trainset)
joblib.dump(classify_model,"classify_0003.m")

## 预测数据
result = classify_model.predict(X_testset)

test_num_0 = 0
test_num_1 = 0
test_num_2 = 0
test_num_3 = 0


predict_num_0 = 0
predict_num_1 = 0
predict_num_2 = 0
predict_num_3 = 0
for i in range(0, result.__len__(),1):
    if y_testset[i] == 0.0:
        test_num_0 += 1
    elif y_testset[i] == 1.0:
        test_num_1 += 1
    elif y_testset[i] == 2.:
        test_num_2 += 1
    else:
        test_num_3 += 1
    if result[i] == y_testset[i]:
        if result[i] == 0.0:
            predict_num_0 += 1
        elif result[i] == 1.0:
            predict_num_1 += 1
        elif result[i] == 2.0:
            predict_num_2 += 1
        else:
            predict_num_3 += 1
    if result[i] != y_testset[i]:
        probe = classify_model.predict_proba(X_testset[i])
        # print probe[0][0], probe[0][1], probe[0][2], probe[0][3],probe[0][4]
        print result[i], y_testset[i]

print result.__len__()
print test_num_1
print 1.0 * predict_num_0 / test_num_0
print 1.0 * predict_num_1 / test_num_1
print 1.0 * predict_num_2 / test_num_2
print 1.0 * predict_num_3 / test_num_3