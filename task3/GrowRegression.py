# coding=utf-8

## 加载数据
from sklearn.datasets import load_svmlight_file
filename = "data/trainingset/valueAttributes.txt"
data = load_svmlight_file(filename)
X, y = data[0], data[1]

## 加载模型
from sklearn.externals import joblib
### 分类
classify_scaler = joblib.load("classify_scaler.m")
classify_model = joblib.load("classify.m")

### 回归模型
big_reg_scaler = joblib.load("big_reg_scaler.m")
big_reg_model = joblib.load("big_reg.m")
small_reg_scaler = joblib.load("small_reg_scaler.m")
small_reg_model = joblib.load("small_reg.m")

## 计算误差
mse = 0.0
for i in range(0,y.__len__(),1):
    one_x = X[i]
    one_classify_scaler_x = classify_scaler.transform(one_x)
    classify_result = classify_model.predict(one_classify_scaler_x)
    predict = 0.0
    if classify_result == 0.:
        small_x = small_reg_scaler.transform(one_x)
        predict = small_reg_model.predict(small_x)
    if classify_result == 1.:
        big_x = big_reg_scaler.transform(one_x)
        predict = big_reg_model.predict(big_x)
    predict = predict[0]
    if predict < 0.0:
        predict = 0.0
    print predict , y[i]
    mse += abs(predict - y[i])
print mse / y.__len__()