#coding=utf-8
import jieba
import os
import numpy as np

from gensim import corpora, models,similarities
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

TRAINING_BLOG = '../newdata/training_document_segment/'
TRAINING_USER_TAG = '../newdata/training.txt'
LABELSPACE = '../newdata/labelSpace.txt'
num_topics = 100


## 对文本进行分词
def segment(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    data = lines[1] + " " + lines[2]
    seg_list = jieba.cut(data)
    newdata = ''
    for word in seg_list:
        newdata += word.encode('utf-8') + ' '
    newdata = newdata.replace('\n', '').strip()
    f2 = open(filename,"w")
    f2.write(newdata)
    f2.close()

## 获得用户-标签空间
def get_user_tag_dict():
    user_tag_dict = {}
    f = open(TRAINING_USER_TAG)
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        tokens = line.split(',')
        user = tokens[0]
        tag_list = [tokens[1],tokens[2],tokens[3]]
        user_tag_dict[user] = tag_list
        # print user, tag_list[0], tag_list[1], tag_list[2]
    return user_tag_dict

## 获取标签空间
def get_labelSpace_dict():
    labelSpace_dict = {}
    f = open(LABELSPACE)
    lines = f.readlines()
    num = 0
    for line in lines:
        label = line.strip()
        if label not in labelSpace_dict:
            labelSpace_dict[label] = num
            num += 1
    return labelSpace_dict

## 获得下标对应的标签
def get_index_label_dict():
    index_label_dict = {}
    labelSpace_dict = get_labelSpace_dict()
    for label in labelSpace_dict:
        index = labelSpace_dict[label]
        index_label_dict[index] = label
    return index_label_dict

## 获得博客-标签空间
def get_blog_tag_dict():
    blog_tag_dict = {}
    user_tag_dict = get_user_tag_dict()
    labelSpace_dict = get_labelSpace_dict()
    for user in user_tag_dict:
        user_blogs_path = TRAINING_BLOG + user
        user_tag_list = user_tag_dict[user]
        user_tag_index_list = [labelSpace_dict[user_tag_list[0]],labelSpace_dict[user_tag_list[1]],labelSpace_dict[user_tag_list[2]]]
        if not os.path.exists(user_blogs_path):
            continue
        for path, dirs, files in os.walk(user_blogs_path):
            for blog in files:
                blog = blog.strip('.txt')
                tag_vector = [0 * i for i in range(0,42,1)]
                for index in user_tag_index_list:
                    tag_vector[index] += 1
                if blog not in blog_tag_dict:
                    blog_tag_dict[blog] = tag_vector
                else:
                    old_tag_vector = blog_tag_dict[blog]
                    for j in range(0,42,1):
                        old_tag_vector[j] += tag_vector[j]
                    blog_tag_dict[blog] = old_tag_vector
    ## blog_tag_dict = {blog_id:[]}
    return blog_tag_dict

## 获得博客的index标记,以及博客内容
def get_blog():
    blog_index_dict = {}
    blog_content_list = []
    index = 0
    user_tag_dict = get_user_tag_dict()
    for user in user_tag_dict:
        user_blogs_path = TRAINING_BLOG + user
        if not os.path.exists(user_blogs_path):
            continue
        for path, dirs, files in os.walk(user_blogs_path):
            for blog in files:
                blog_content_file = user_blogs_path + '/' + blog

                ## 训练才用到博文内容
                # f = open(blog_content_file)
                # data = f.read().strip()
                # # print data
                # blog_content_list.append(data)

                blog = blog.strip('.txt')
                blog_index_dict[index] = blog
                index += 1

    return blog_index_dict,blog_content_list

## 获取博客-标签矩阵
def get_blog_tag_matrix(blog_index_dict, blog_tag_dict):
    blog_tag_matrix = []
    length = blog_index_dict.__len__()
    for i in range(0, length, 1):

        ## 先获得索引对应的博客id，然后在id基础上找到博客的向量
        blog_id = blog_index_dict[i]
        blog_tag_vector = blog_tag_dict[blog_id]

        ## 做归一化处理
        max_num = 0
        for j in range(0, blog_tag_vector.__len__(), 1):
            if blog_tag_vector[j] > max_num:
                max_num = blog_tag_vector[j]
        if max_num > 0:
            for j in range(0, blog_tag_vector.__len__(),1):
                blog_tag_vector[j] = 1.0 * blog_tag_vector[j] / max_num

        blog_tag_matrix.append(blog_tag_vector)
    blog_tag_matrix = np.array(blog_tag_matrix)
    return blog_tag_matrix


## 训练lsi模型
def train_lsi_model(blog_index_dict,blog_content_list):
    documents = blog_content_list
    texts = [[word for word in document.lower().split()] for document in documents]
    dictonary = corpora.Dictionary(texts)
    dictonary.save('../dictionary.model')
    corpus = [dictonary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    tfidf.save('../tfidf.model')
    corpus_tfidf = tfidf[corpus]
    corpus_tfidf.save('../corpus_tfidf.model')
    lsi = models.LsiModel(corpus_tfidf, id2word=dictonary,num_topics=num_topics)
    lsi.save('../lsi.model')
    corpus_lsi = lsi[corpus_tfidf]
    corpus_lsi.save('../corpus_lsi.model')
    index = similarities.MatrixSimilarity(lsi[corpus])
    index.save('../index.model')

## 构建主题-标签模型
def get_topic_label_matrix(blog_tag_matrix):

    ## 获取博客-主题矩阵
    blog_topics_matrix = []
    corpus_lsi = corpora.MmCorpus.load('../corpus_lsi.model')  ## 读取模型
    for i in range(0, corpus_lsi.__len__(), 1):
        doc = corpus_lsi[i]
        topic_vector = [0* j for j in range(num_topics)]
        for k in range(0, num_topics, 1):
            topic_vector[k] = doc[k][1]
        blog_topics_matrix.append(topic_vector)

    ## 将博客-主题矩阵转置, 得到主题-博客矩阵
    blog_topics_matrix = np.array(blog_topics_matrix)
    topic_blog_matrix_T = blog_topics_matrix.T

    ## 将主题-博客矩阵与博客-标签矩阵相乘, 得到主题-标签的特征矩阵
    topic_tag_matrix = np.dot(topic_blog_matrix_T, blog_tag_matrix)
    for i in range(0, topic_tag_matrix.__len__(),1):
        print topic_tag_matrix[i]

    ## 保存矩阵
    np.save('../topic_tag.npy', topic_tag_matrix)
    return topic_tag_matrix


## 获取用户标签向量矩阵
def get_user_tag_vector(user):

    ## 读取主题-标签矩阵
    topic_tag_matrix = np.load('../topic_tag.npy')

    ## 读取字典
    dictonary = corpora.Dictionary.load('../dictionary.model')

    ## 读取lsi模型
    lsi = models.LsiModel.load('../lsi.model')

    ## 将每篇博客映射成博客-主题向量，然后相加
    user_topic = np.array([0.0*i for i in range(0,num_topics,1)])
    user_blogs_path = TRAINING_BLOG + user
    if not os.path.exists(user_blogs_path):
        return []
    for path, dirs, files in os.walk(user_blogs_path):
        for blog in files:
            blog_content_file = user_blogs_path + '/' + blog
            blog_content = open(blog_content_file).read().strip()
            query_bow = dictonary.doc2bow(blog_content.lower().split())
            query_lsi = lsi[query_bow]
            query_vector = [0*i for i in range(0,num_topics,1)]
            for j in range(0,num_topics,1):
                query_vector[j] = query_lsi[j][1]
            query_vector = np.array(query_vector)
            user_topic += query_vector

    ## 将用户-主题向量 与 主题-标签矩阵相乘， 得到用户-标签向量
    user_tag_vector = np.dot(user_topic, topic_tag_matrix)
    return user_tag_vector



# 预测用户标签前所做的工作
## 读取字典
dictonary = corpora.Dictionary.load('../dictionary.model')
## 读取lsi模型
lsi = models.LsiModel.load('../lsi.model')
## 读取index模型
index = similarities.MatrixSimilarity.load('../index.model')
## 得到标签的下标
index_label_dict = get_index_label_dict()


## 获取用户标签向量矩阵
def get_user_similarity_tag(user, blog_index_dict, blog_tag_dict):

    ## 选择最相似的博客数
    sim_k = 5

    ## 用户的预测标签向量
    user_tag_vector = [0* t for t in range(0,42,1)]

    ## 获取每篇博客最相似的博客，以其标签做协同
    user_blogs_path = TRAINING_BLOG + user
    if not os.path.exists(user_blogs_path):
        return ['移动开发','软件工程','web开发']
    for path, dirs, files in os.walk(user_blogs_path):
        for blog in files:
            blog_vector = np.array([0 * b for b in range(0,42,1)])
            blog_content_file = user_blogs_path + '/' + blog
            blog_content = open(blog_content_file).read().strip()
            query_bow = dictonary.doc2bow(blog_content.lower().split())
            query_lsi = lsi[query_bow]
            sims = index[query_lsi]
            sorted_sims = sorted(enumerate(sims),key=lambda item:-item[1])

            ##找最相似的前k篇博客
            for i in range(0,sim_k,1):
                blog_index = sorted_sims[i][0]
                blog_id = blog_index_dict[blog_index]
                blog_tag_vector = np.array(blog_tag_dict[blog_id])
                blog_vector += blog_tag_vector

            ##k篇博客中找到最相似的标签
            k_max = blog_vector[np.argpartition(blog_vector, -3)[-3:]]
            for i in range(0,blog_vector.__len__(),1):
                if blog_vector[i] in k_max:
                    user_tag_vector[i] += 1

    # print user_tag_vector
    user_tag_vector = np.array(user_tag_vector)
    sim_max = user_tag_vector[np.argpartition(user_tag_vector, -3)[-3:]][::-1]
    user_label = []
    for j in range(0,user_tag_vector.__len__(),1):
        if user_tag_vector[j] in sim_max:
            label = index_label_dict[j]
            user_label.append(label)
            if user_label.__len__() >= 3:
                break
    return user_label



blog_tag_dict = get_blog_tag_dict() ## 获得博客--标签映射表
blog_index_dict,blog_content_list = get_blog() ## 获取博客-id映射表，以及博客内容列表
# train_lsi_model(blog_index_dict,blog_content_list)
# blog_tag_matrix = get_blog_tag_matrix(blog_index_dict, blog_tag_dict) ## 获取博客-标签矩阵
# get_topic_label_matrix(blog_tag_matrix)  ## 获取主题-标签矩阵

precision = 0.0
num = 0
user_tag_dict = get_user_tag_dict()
for user in user_tag_dict:
    user_predict = get_user_similarity_tag(user, blog_index_dict, blog_tag_dict)
    user_actual = user_tag_dict[user]
    print user
    print 'predict',
    for label in user_predict:
        print label,
    print
    print 'actual',
    for label in user_actual:
        print label,
    print
    list_inter = list(set(user_predict).intersection(set(user_actual)))
    precision += 1.0 * list_inter.__len__() / 3
    num += 1


precision = precision / num
print precision
# labelSpace_dict = get_labelSpace_dict()
# for label in labelSpace_dict:
#     print label, labelSpace_dict[label]



























