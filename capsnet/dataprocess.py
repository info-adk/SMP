# coding:utf-8
import os
import numpy as np

#binary to str, delete blank line
def coding():
    datapath = "/home/iiip/Info/capsnet/data/20_newsgroups"
    new_datapath = "/home/iiip/Info/capsnet/data/20_newsgroups_new"
    for labelname in os.listdir(datapath):
        labeldir = os.path.join(datapath,labelname)
        new_labeldir = os.path.join(new_datapath,labelname)
        if not os.path.exists(new_labeldir):
            os.mkdir(new_labeldir)
        for i in os.listdir(labeldir):
            print(i)
            text = os.path.join(labeldir,i)
            new_text = open(os.path.join(new_labeldir,i),"w")
            content = ""
            with open(text,"rb") as f:
                lines = f.readlines()
                blank = lines.index(b"\n",0) #找到第一个空白行
                for item in lines[blank+1:]: #获取空白行下一行到末尾部分
                    item = item.decode("ISO-8859-2")
                    if item != "\n" and item.strip() !="":
                        content+=item.strip()+" "
            content = content.strip()
            new_text.write(content)
            new_text.close()
coding()
