# coding=utf-8

import gensim
import numpy as np
import os
from gensim.models.doc2vec import Doc2Vec,LabeledSentence

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_trainset():
    x_train = []
    dirs = 'data/blogfenci'
    num = 0
    index = 0
    doc_dict = {}
    for path, dirname, filelist in os.walk(dirs):
        for filename in filelist:
            if num < 19001:
                num += 1
                continue
            user_file = dirs + '/' +  filename
            data = open(user_file).read()
            data = data.replace('\n',  " ").replace('  ', ' ')
            data = data.lower()
            words = data.split(" ")
            x_train.append(TaggededDocument(words, tags=[index]))
            # print filename.replace(".txt","")
            doc_dict[index] = filename.replace(".txt", "")
            index += 1
            num += 1
            print num
            if num > 20001:
                break
    return x_train, doc_dict

## 获取向量
def getVecs(model,corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1,size)) for z in corpus]
    return np.concatenate(vecs)

def train(x_train, size=500, epoch_num=1):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model/doc2vecmodel20')
    return model_dm

def demo():
    model_dm = Doc2Vec.load("doc2vec/doc2vecmodel")
    test_text = open('data/blogfenci/D0650192.txt').read().split(" ")
    inferred_vector_dm = model_dm.infer_vector(test_text)
    # print inferred_vector_dm
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)

    return sims

def save_doc_index(doc_dict):
    index_file = "model/index_file20.txt"
    lines = ""
    for index in doc_dict:
        lines += str(index) + ' ' + doc_dict[index] + '\n'
    f = open(index_file, 'w')
    f.write(lines)
    f.close()

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


if __name__ == '__main__':
    # user_tags_dic = load_trainset()
    # x_train, doc_dict = get_trainset()
    # save_doc_index(doc_dict)
    # doc_dict = load_doc_index()
    # model_dm = train(x_train)
    # print 'nice'
    # sim_list = []
    # total_dict = {}
    # sims = demo()
    # print 'well'

    doc_dict = load_doc_index()
    sims = demo()
    for index, sim in sims:
        print sim, doc_dict[index]


    # num = 0
    # for index, sim in sims:
    #     if num == 0:
    #         num = 1
    #         continue
    #     print doc_dict[index],
    #     sim_user = doc_dict[index]
    #     sim_list.append(sim_user)
    #     tag_list = user_tags_dic[sim_user]
    #     for tag in tag_list:
    #         if tag not in total_dict:
    #             total_dict[tag] = 0
    #         total_dict[tag] += 1
    # sort_dict = sorted(total_dict.items(), key=lambda item: -item[1])
    # print
    # for tag in sort_dict:
    #     print tag[0], tag[1]
        # print sim
        # doc = x_train[int(index)]
        # doc = doc[0]
        # words = ''
        # for word in doc:
        #    print word,
        # print
        # print sim
        # print len(doc[0])








