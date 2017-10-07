# coding=utf-8

## 保存模型
from sklearn.externals import joblib

## 加载数据
from sklearn.datasets import load_svmlight_file
filename = "data/trainingset/valueMostAttributes.txt"
data = load_svmlight_file(filename)
X, y = data[0], data[1]
X = X.toarray()

## 数据处理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
# scaler = joblib.load("reg_scaler_0003-001.m")
X = scaler.fit(X).transform(X,y)
joblib.dump(scaler, "reg_scaler_0003-001.m")

# from sklearn.feature_selection import SelectKBest, f_regression
# X = SelectKBest(f_regression, k=300).fit(X,y)


## 划分数据集
from sklearn.model_selection import ShuffleSplit
rs = ShuffleSplit(n_splits=1,train_size=0.7, test_size=0.3, random_state=1)
rs.get_n_splits(X)
X_trainset = None
y_trainset = None
X_testset = None
y_testset = None

for train_index, test_index in rs.split(X,y):
    X_trainset, X_testset = X[train_index], X[test_index]
    y_trainset, y_testset = y[train_index], y[test_index]

# ## 模型训练
from sklearn.linear_model import BayesianRidge,HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import  BaggingRegressor
regression_model = BayesianRidge()
regression_model.fit(X_trainset,y_trainset)
#
# bagging = BaggingRegressor(BayesianRidge(),n_estimators=10)
# bagging = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=3, random_state=0, loss='ls')
# bagging.fit(X_trainset, y_trainset)
# regression_model = bagging
# joblib.dump(regression_model, "reg_0003-001.m")

## 预测测试集
X_testset = X
y_testset = y
lines = ""
# regression_model = joblib.load("reg_0003-001.m")
result = regression_model.predict(X_testset)
mse = 0.0
for i in range(0, result.__len__(), 1):
    if result[i] < 0:
        result[i] = 0
    print round(result[i],4), y_testset[i]
    mse = abs(result[i]- y_testset[i])
    lines += str(round(result[i],4)) + "\n"
# print lines
print round(mse / result.__len__(),7)
