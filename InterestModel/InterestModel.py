# coding=utf-8

import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

## 读取训练数据
TRAINING_DATA = '../newdata/train_lsi.txt'
TRAINING_USER_TAG = '../newdata/training.txt'
LABELSPACE = '../newdata/labelSpace.txt'
VALIDATION_USER_TAG = '../newdata/validation.txt'
VALIDATE_DATA = '../newdata/vali_lsi.txt'

## 获取标签空间
def get_labelSpace_dict():
    labelSpace_dict = {}
    label_list = []
    f = open(LABELSPACE)
    lines = f.readlines()
    num = 0
    for line in lines:
        label = line.strip()
        if label not in labelSpace_dict:
            labelSpace_dict[label] = num
            num += 1
            label_list.append(label)
    return labelSpace_dict, label_list


## 获得用户-标签空间
def get_training_user_tag_dict():
    user_tag_dict = {}
    f = open(TRAINING_USER_TAG)
    # f = open(VALIDATION_USER_TAG)
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        tokens = line.split(',')
        user = tokens[0]
        tag_list = [tokens[1],tokens[2],tokens[3]]
        user_tag_dict[user] = tag_list
        # print user, tag_list[0], tag_list[1], tag_list[2]
    return user_tag_dict

## 获得用户-标签空间
def get_vali_user_tag_dict():
    user_tag_dict = {}
    f = open(VALIDATION_USER_TAG)
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        tokens = line.split(',')
        user = tokens[0]
        tag_list = [tokens[1],tokens[2],tokens[3]]
        user_tag_dict[user] = tag_list
    return user_tag_dict

labelSpace_dict, label_list = get_labelSpace_dict()
training_user_tag_dict = get_training_user_tag_dict()
vali_user_tag_dict = get_vali_user_tag_dict()

## 将标签转化为向量空间
def label_to_vector(user):
    label_vector = [0 * i for i in range(0,42,1)]
    user_label = training_user_tag_dict[user]
    for label in user_label:
        j = labelSpace_dict[label]
        label_vector[j] = 1
    return label_vector


def load_training_data():
    f = open(TRAINING_DATA)
    lines = f.readlines()
    X = []
    y = []
    for line in lines:
        line = line.strip()
        tokens = line.split(" ")
        user = tokens[0]
        features = tokens[1:]
        for i in range(0,features.__len__(),1):
            features[i] = float(features[i])
        labels = training_user_tag_dict[user]
        for label in labels:
            X.append(features)
            y.append(labelSpace_dict[label])
    return X, y

def load_data():
    f = open(VALIDATE_DATA)
    lines = f.readlines()
    X = []
    y = []
    for line in lines:
        line = line.strip()
        tokens = line.split(" ")
        user = tokens[0]
        features = tokens[1:]
        for i in range(0,features.__len__(),1):
            features[i] = float(features[i])
        labels = vali_user_tag_dict[user]
        label_y = []
        for label in labels:
            label_y.append(labelSpace_dict[label])
        X.append(features)
        y.append(label_y)
    return X, y



import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

## one_hot映射
def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

def train(X, y):  ## X为特征向量, y为目标属性

    train_y_ohe = one_hot_encode_object_array(y)
    model = Sequential()
    model.add(Dense(60, input_shape=(200,)))
    model.add(Activation('sigmoid'))
    # model.add(Dropout(0.02))  ## 避免过拟合
    model.add(Dense(42))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(X, train_y_ohe, nb_epoch=100, batch_size=1, verbose=1)
    return model


def predict(X, model):
    predict_y = []
    result = model.predict_proba(X) ## 预测概率
    for i in range(0,result.__len__(), 1):
        result[i] = np.array(result[i])
        topk = result[i][np.argpartition(result[i],-3)[-3:]] ## 取概率最大的前三个标签
        predict = []
        for j in range(0,42,1):
            if result[i][j] in topk:
                predict.append(j)
        predict_y.append(predict)
    return predict_y


## 训练模型
X, y = load_training_data()
step2_model = train(X, y)

X_test, y_test = load_data()
step1_X = X_test
predict_y = predict(step1_X,step2_model)
precision = 0.0
num = 0
for i in range(0,y_test.__len__()):
    print y_test[i], predict_y[i]
    list_inter = list(set(y_test[i]).intersection(set(predict_y[i])))
    precision += 1.0 * list_inter.__len__() / 3
    num += 1
print precision / num