# -*- coding: utf-8 -*-
"""
Created on 2018/4/28 12:39

@author: Eric

Summary: 

    After the training process in classifiertrain.py, i get the optimal paremeters for thread detection:
w_contextfree = 0.14， threshold = 0.6, w_auther=0.3, w_conver=0.6, w_temp=0.1, fvalue = 0.624
auther_scale = 2.347 conver_scale = 0.383 temporal_scale = 4.686

Run order: 2.( After classifiertrain.py)
"""
#%%
import re
import pandas as pd
from pyltp import SentenceSplitter
from classifier_base_functions import ExpandDF
#%%
# read from pickle, the chat log from 2017/05/06 - 2018/02/28
total_msg = pd.read_pickle(r'../records/sample/sample_msg_pickle')
total_msg.dropna(subset = ['msg'], how = 'any', inplace = True)# 637255
#total_msg = total_msg[-total_msg['msg'].str.contains(r'<msg>')]#556777
#total_msg = total_msg[-total_msg['msg'].str.contains(r'复制这条信息')]#556777
total_msg = total_msg[-total_msg['msg'].str.contains(r'img aeskey')]#556777
# get app info title
re_app_title = re.compile(r'<title>(?P<apptitle>.*?)</title>', re.S)
apptitledf = pd.Series(total_msg['msg'].str.findall(re_app_title),name = 'msg')
apptitledf = pd.Series(map(lambda x: None if len(x) == 0 else x[0], apptitledf),name = 'msg',index = apptitledf.index)
apptitledf.dropna(inplace = True)
total_msg['msg'].update(apptitledf)
#filter at wechatid
re_filter_msgat = re.compile(u'@(?P<atname>.*?)\u2005')
total_msg['msg'] = total_msg['msg'].str.replace(re_filter_msgat,'')#replace the @wechatid with '', this should be done when deal with predict dataset.
#filter \n------------
re_copyThisMsg = re.compile(r'------------.*$',re.S)#re.sub(re_copyThisMsg, '', total_msg.msg[5])
total_msg['msg'] = total_msg['msg'].str.replace(re_copyThisMsg,'')
#%% initilize paremeters
twindow = 0.5#这个应该用在全局的totalmsg表上，每次处理响应窗口的数据
divideN = 1500#对timediff进行缩放

threshold = 0.6
w_contextfree = 0.14
w_auther=0.3
w_conver=0.6
w_temp=0.1

auther_scale = 2.347
conver_scale = 0.383
temporal_scale = 4.686

#%% split words

import os
LTP_DATA_DIR = r'D://ltp_data//ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

#在此基础上加入分词的语句
from pyltp import Segmentor
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型

wordsLists = []
for index, msg in total_msg['msg'].iteritems():
    #print(msg)
    sents = SentenceSplitter.split(msg)
    #sentsLists.append(sents)
    wordslist = []
    for sent in sents:
        #过滤句子中的标点符号
        words = segmentor.segment(sent)
        wordslist.extend(list(words))
    wordsLists.append(wordslist)
total_msg['splitwords'] = wordsLists

segmentor.release()  # 释放模型,分词完毕

#%% Build the BOW
from gensim import corpora, models
from six import iteritems
dictionary = corpora.Dictionary(total_msg.splitwords.tolist())
#read stopwords list
with open(r'stopwords.tab', encoding = 'utf-8') as f_stop:
     stopwords = [line.strip() for line in f_stop]
#remove stop words and words appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id]

once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]

dictionary.filter_tokens(stop_ids + once_ids)
dictionary.compactify()# remove gaps in id sequence after words that were removed

#initialize the bag of words model
corpus = [dictionary.doc2bow(text) for text in total_msg['splitwords'].tolist()]
tfidf = models.TfidfModel(corpus)
corpus_tfidf =  tfidf[corpus]

total_msg['tfidf'] = list(corpus_tfidf)
#ann2['tfidf'] = corpus
total_msg['tfidf'] = list(map(lambda x: dict(x), total_msg['tfidf']))# to dict
#data persistence
total_msg.to_pickle(r'../records/sample/sample_msg_tfidf_pickle')

#%% Expand msg
"""NOW I CAN SKIP CODES ABOVE AND EXECUTE CODES BELOW
"""
msgat = pd.read_pickle(r'../records/sample/msgat_dataframe')
total_msg = pd.read_pickle(r'../records/sample/sample_msg_tfidf_pickle')
#TODO
ExpandDF(total_msg,msgat,w_contextfree,w_auther,w_contextfree,w_temp)
