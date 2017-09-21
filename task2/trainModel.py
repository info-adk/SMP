# coding=utf-8

relate_dict = {'移动开发':['人机交互','桌面开发'],
               '机器学习':['人工智能','深度学习'],
               '人脸识别':['计算机视觉', '人工智能'],
               '硬件':['嵌入式开发']}

blog_act_weight = {'post':0.15, 'favorite':0.15, 'comment':0.15, 'vote':0.15, 'browse':0.1}

import  math
## 读取标签空间
def load_labelSpace():
    labelSpace_list = []
    f = open('data/newdata/label/labelSpace.txt')
    lines = f.readlines()
    for i in range(0, len(lines), 1):
        label = lines[i].strip()
        labelSpace_list.append(label)
    return labelSpace_list

from gensim.models.doc2vec import Doc2Vec

def load_doc_index():
    doc_dict = {}
    index_file = "doc2vec/index_file.txt"
    f = open(index_file)
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.strip()
        tokens = line.split(" ")
        doc_dict[int(tokens[0])] = tokens[1]
    return doc_dict

doc_dict = load_doc_index()
model_dm = Doc2Vec.load("doc2vec/doc2vecmodel")

def doc2vec_sim(blog_id):
    test_text = open('data/blogfenci/' + blog_id + '.txt').read().split(" ")
    inferred_vector_dm = model_dm.infer_vector(test_text)
    # print inferred_vector_dm
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    sim_tag_dict = {}
    for index, sim in sims:
        if sim < 0.25:
            break
        else:
            sim_tag = doc_dict[index]
            if sim_tag not in sim_tag_dict:
                sim_tag_dict[sim_tag] = 0
            sim_tag_dict[sim_tag] += 1
    sim_tag_dict = sorted(sim_tag_dict.items(), key=lambda item:-item[1])
    sim_tag_list = []
    if len(sim_tag_dict) < 3:
        for tag in sim_tag_dict:
            sim_tag_list.append(tag[0])
    else:
        sim_tag_list = [sim_tag_dict[0][0], sim_tag_dict[1][0],sim_tag_dict[2][0]]

    return sim_tag_list


## 读取用户标签频率
def load_labelFre():
    labelFre_dic = {}
    f = open("data/newdata/label/labelFrequency.txt")
    lines = f.readlines()
    for i in range(0, len(lines), 1):
        line = lines[i].strip()
        tokens = line.split(" ")
        label, fre = tokens[0], int(tokens[1])
        labelFre_dic[label] = fre
        # print label ,fre
    return labelFre_dic

## 读取用户标签
def load_trainset():
    user_tags_dic = {}
    filename = "data/training.txt"
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip()
        tokens = line.split(",")
        user = tokens[0]
        tag1, tag2, tag3 = tokens[1], tokens[2], tokens[3]
        tag_list =[tag1,tag2,tag3]
        user_tags_dic[user] = tag_list
    return user_tags_dic


## 读取训练样本, 返回样本词语集, 以及每一个样本对应的用户
import os
def load_traingCorpus():
    dir_name = 'data/newdata/trainingCorpus/'
    docs_training = []
    doc_user = {} ## 每一个doc对应的用户
    index = 0
    for path, dirname, filelist in os.walk(dir_name):
        for file_name in filelist:
            user_file = dir_name + '/' + file_name
            line = open(user_file).read().strip()
            line = line.lower()  ## 转化为小写
            if line.__len__() > 150:  ## 除去一些稀疏的用户
                word_list = line.split(" ")
                docs_training.append(word_list)
                doc_user[index] = file_name.strip(".txt")
                index += 1
            else:
                print line
    return docs_training, doc_user

def blog_tag(blog_id, keyword_dict):
    blog_dir = 'data/blog/'
    keyword_blog_score = {}
    f = open(blog_dir + blog_id + '.txt')
    data = f.read().lower()
    blog_str = data[0:100]
    for label in keyword_dict:
        keyword_blog_score[label] = 0
        keyword_list = keyword_dict[label]
        for word in keyword_list:
            if word == '':
                continue
            if word in data:
                if isEnglish(word):
                    groups2 = re.findall('[A-Za-z]' + word, data)
                    groups3 = re.findall(word + '[A-Za-z]', data)
                    if len(groups2) != 0 and len(groups3) != 0:
                        continue
            num = data.count(word)
            if num >= 10:
                num = 9
            keyword_blog_score[label] += num
    keyword_blog_score = sorted(keyword_blog_score.items(), key=lambda item:-item[1])
    blog_tag_list = []
    for key in keyword_blog_score:
        if key[1] > 0:
            blog_tag_list.append(key[0])
    return blog_tag_list,blog_str

def blog_act(user_id):
    blog_act_dir = 'data/userBlog/'
    f = open(blog_act_dir + user_id + '.txt')
    blog_act_dict = {}
    blog_act_list = f.readlines()
    blog_act_dict['post'] = blog_act_list[0]
    blog_act_dict['favorite'] = blog_act_list[1]
    blog_act_dict['comment'] = blog_act_list[2]
    blog_act_dict['vote'] = blog_act_list[3]
    blog_act_dict['browse'] = blog_act_list[4]
    return blog_act_dict

import re
## 判断是否英文
def isEnglish(test_str):
    groups = re.findall('[A-Za-z]', test_str)
    if len(groups) == len(test_str):
        return True
    else:
        return False

## 基础工作
labelSpace_list = load_labelSpace() ## 读取标签空间
labelFre_dic = load_labelFre()  ## 读取标签频率词典


precision = 0
num = 0
## step 1: 读取关键词典, 字母转为小写
keyword_dict = {}
keyword_file_dir = 'data/newdata/label/KeyWords2'
for path, dirname, filelist in os.walk(keyword_file_dir):
    for filename in filelist:
        keyword_file = keyword_file_dir + '/' + filename
        data = open(keyword_file).read()
        data = data.lower()
        keywords = data.split(" ")
        label = filename.replace(".txt","")
        keyword_dict[label] = keywords

## step 2: 构建LSI模型, 将用户的所有标题用作训练样本
import logging
import jieba
from gensim import corpora, models , similarities
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level = logging.INFO)
user_tags_dic = load_trainset()  ## 读取用户以及实际标签
docs_traing, doc_user = load_traingCorpus()  ## 读取训练用户的语料, 作为训练lsi
dictionary = corpora.Dictionary(docs_traing)
corpus = [dictionary.doc2bow(doc) for doc in docs_traing]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=250)  ##隐藏话题为42个
# lsi.save( '/media/qistar/DATA/nlp/lsimodel/model-7.lsi' )
# lsi = models.LsiModel.load('/media/qistar/DATA/nlp/lsimodel/model-1.lsi')
index = similarities.MatrixSimilarity(lsi[corpus])

## step 3: 对于每个用户, 取每一个标题, 计算lsi相似度, 设定阈值
## 读取验证集用户
#
file_dir = 'data/trainingTitle/'
for path, dirname, filelist in os.walk(file_dir):
    for filename in filelist:
        # filename = 'U0155594.txt'
#         print filename#
# #
        ## 初始化不同权重得分
        # print user_file
        total_score = {}
        keyword_score = {}  ## 初始化每一个tag的关键词得分, 设为0
        sim_tag_score = {}  ## 初始化每一个tag的相似度得分, 设为0
        relate_score = {}
        relate_3_score = {}
        hot_score = {}
        for label in labelSpace_list:
            keyword_score[label] = 0
            sim_tag_score[label] = 0
            total_score[label] = 0
            hot_score[label] = 0


        ## 读取文件
        user_file = file_dir + filename
        user = filename.replace(".txt", "")
        titles = open(user_file).readlines()
        total_tag_dict = {}
        blog_act_dict = blog_act(user)
        for title in titles:
            title = title.strip()
            blog_id = title.split("`")[0]
            title = title.split("`")[1].lower()
            # print blog_id

            ## 为浏览行为添加权重
            weight = 0
            for act in blog_act_dict:
                if blog_id in blog_act_dict[act]:
                    weight = blog_act_weight[act]
                    break

            ## 检查每一个词是否在关键词中出现
            flag = 0
            for label in keyword_dict:
                keyword_list = keyword_dict[label]
                for word in keyword_list:
                    if word == '':
                        continue
                    if word in title:
                        if isEnglish(word):
                            groups2 = re.findall('[A-Za-z]' + word, title)
                            groups3 = re.findall(word + '[A-Za-z]', title)
                            if len(groups2) != 0 and len(groups3) != 0:
                                continue
                        keyword_score[label] += 1 * weight
                        flag = 1
                        break

            ## doc2vec模型
            if flag == 0:
                sim_tag_list = doc2vec_sim(blog_id)
                for sim_tag in sim_tag_list:
                    keyword_score[sim_tag] += 0.3 * weight
            # if flag == 1:
            #     sim_tag_list = doc2vec_sim(blog_id)
            #     for sim_tag in sim_tag_list:
            #         keyword_score[sim_tag] += 0.2 * weight

            ## 检查每一篇博客关键词的出现情况
            blog_tag_list, blog_str=blog_tag(blog_id, keyword_dict)
            for label in keyword_score:
                if label in blog_tag_list:
                    keyword_score[label] += 0.1 * weight

            ## 计算lsi相似度
            # if flag == 0:
            #     title = blog_str
            seg_list = jieba.cut(title)
            doc = []
            for word in seg_list:
                if word.__len__() >= 2:
                    doc.append(word.encode('utf-8'))
            # for key in blog_keyword_list:
            #     doc.append(key)
            query_bow = dictionary.doc2bow(doc)
            query_lsi = lsi[query_bow]
            sims = index[query_lsi]
            sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
            if sort_sims[0][1] <= 0.3:  ## 将相似度低于0.3的标题过滤掉
                continue
            user_list = []
            tag_list = []
            tag_dict = {}
            for i in range(0,15,1):
                user2 = doc_user[sort_sims[i][0]]
                user_list.append(user2)
                tag_list.append(user_tags_dic[user2])
            for li in tag_list:
                for tag in li:
                    if tag not in tag_dict:
                        tag_dict[tag] = 1
                    else:
                        tag_dict[tag] += 1
            tag_dict = sorted(tag_dict.items(), key=lambda item: -item[1])
            for i in range(0,3,1):
                tag = tag_dict[i][0]
                if tag not in total_tag_dict:
                    total_tag_dict[tag] = 1
                else:
                    total_tag_dict[tag] += 1

        ## 相似度的标签排名
        for tag in total_tag_dict:
            sim_tag_score[tag] = total_tag_dict[tag]

        ## 融合关键词与相似度的排名
        for tag in total_score:
            total_score[tag] = keyword_score[tag] * 2 + sim_tag_score[tag] * 0.1

        total_score = sorted(total_score.items(), key=lambda item: -item[1])

        ## 添加相关联标签的权重
        resorted_score = {}
        for tag in total_score:
            resorted_score[tag[0]] = tag[1]
        if total_score[0][0] == '移动开发' or total_score[1][0] == '移动开发' or total_score[2][0] == '移动开发':
            resorted_score['人机交互'] += resorted_score['移动开发'] * 0.01
        if total_score[0][0] == '移动开发' or total_score[1][0] == '移动开发' or total_score[2][0] == '移动开发':
            resorted_score['桌面开发'] += resorted_score['移动开发'] * 0.01
        if total_score[0][0] == '硬件' or total_score[1][0] == '硬件' or total_score[2][0] == '硬件':
            resorted_score['嵌入式开发'] += resorted_score['硬件'] * 0.3
        if total_score[0][0] == '计算机视觉' or total_score[1][0] == '计算机视觉':
            resorted_score['人脸识别'] += resorted_score['计算机视觉'] * 0.1
        if total_score[0][0] == '计算机视觉' or total_score[1][0] == '计算机视觉':
            resorted_score['图像处理'] += resorted_score['计算机视觉'] * 0.1
        if total_score[0][0] == '图像处理' or total_score[1][0] == '图像处理':
            resorted_score['计算机视觉'] += resorted_score['图像处理'] * 0.2
        if total_score[0][0] == '网络与通信' or total_score[1][0] == '网络与通信' or total_score[2][0] == '网络与通信':
            resorted_score['网络管理与维护'] += resorted_score['网络与通信'] * 0.2
        if total_score[0][0] == '网络管理与维护' or total_score[1][0] == '网络管理与维护' or total_score[2][0] == '网络管理与维护':
            resorted_score['网络与通信'] += resorted_score['网络管理与维护'] * 0.2
        if total_score[0][0] == '机器学习' or total_score[1][0] == '机器学习':
            resorted_score['人工智能'] += resorted_score['机器学习'] * 0.1

        # if total_score[0][0] == '软件工程' or total_score[1][0] == '软件工程':
        resorted_score['信息安全'] += resorted_score['网络与通信'] * 0.1 + resorted_score['网络管理与维护'] * 0.1

        total_score = sorted(resorted_score.items(), key=lambda item: -item[1])
        predict_tag = []
        for i in range(0, 3, 1):
            # print total_score[i][0],total_score[i][1]
            predict_tag.append(total_score[i][0])
        if total_score[0][1] == 0.0:
            predict_tag = ['移动开发','软件工程','web开发']
        actual_tag = user_tags_dic[user]
        list_inter = list(set(predict_tag).intersection(set(actual_tag)))
        print user,
        print 'predict'
        for tag in predict_tag:
            print tag,
        print
        print 'actual'
        for tag in actual_tag:
            print tag,
        print
        precision += 1.0 * list_inter.__len__() / 3
        num += 1
print precision / num