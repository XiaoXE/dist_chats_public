# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:47:31 2018

@author: Eric

run order:1

"""
#%%
import scipy.stats as st
import matplotlib.pyplot as plt
import nltk
#nltk.download()
from pyltp import SentenceSplitter
#%%

#%%
'''
feature building
1.context info of message: author context, conversational context, temperal context
2.lexical info of message: discourse(cue words, question, long) content(repeat)

We used the development set to approximate the normal densities used in our context models and the evaluation set to obtain
the results reported below.
'''
ann1done = pd.read_csv(r'../records/annotation/anno1done.csv')
ann2done = pd.read_csv(r'../records/annotation/anno2done.csv')

ann1 = pd.read_csv(r'../records/annotation/anno1.csv')
ann2 = pd.read_csv(r'../records/annotation/anno2.csv')

ann1 = pd.concat([ann1,ann1done[['thread','appthread']]],axis  = 1)
del ann1done
ann2 = pd.concat([ann2,ann2done[['thread','appthread']]],axis  = 1)
del ann2done

ann1.createTime = pd.to_datetime(ann1.createTime)
ann2.createTime = pd.to_datetime(ann2.createTime)

ann1.dropna(subset = ['msg','thread'],how = 'any', inplace = True)
ann2.dropna(subset = ['msg','thread'],how = 'any', inplace = True)

ann1 = ann1[ann1.thread > -2]
ann2 = ann2[ann2.thread > -2]
#set the time window 群组讨论的当天和前后半天
twindow = 0.5#这个应该用在全局的totalmsg表上，每次处理响应窗口的数据
#%%
#%%
'''
the author context of a message m, denoted by CA(m), is the set of other messages
written by m’s author am:
'''
#context info of message
#author context
    
totalTimediff = []
for row in ann1.itertuples():
    msgid = row[2]
    sender = row[6]
    msgdate = row[4]
    maxdate = msgdate + datetime.timedelta(days = 1)
    mindate = msgdate - datetime.timedelta(days = 1)
    threadid = row[10] 
    autherContext = ann1[(ann1.sender == sender)&(ann1.msgSvrId != msgid)]
    timediff = autherContext[autherContext.thread == threadid]['createTime'] - msgdate
    timediff = timediff.tolist()#单位秒
    timediff = [x.total_seconds() for x in timediff]
    totalTimediff.extend(timediff)

#估计这个timediff的参数
plt.hist(totalTimediff, normed = True)
st.norm.fit(totalTimediff)#loc = 0 scale = 3520
#%%
#%%
'''
conversational context 
'''
totalTimediff = []
for row in ann1.itertuples():
    msgid = row[2]
    sender = row[6]
    msgdate = row[4]
    maxdate = msgdate + datetime.timedelta(days = 1)
    mindate = msgdate - datetime.timedelta(days = 1)
    threadid = row[10]
    #我提到了那些人
    mentionNames = msgat[(msgat.msgSvrId == msgid)&(msgat.createTime > mindate)&(msgat.createTime < maxdate)]['member_x'].tolist()
    #我被那些人提到
    mentionNames.extend(msgat[(msgat.member_x == sender)&(msgat.createTime > mindate)&(msgat.createTime < maxdate)]['sender'].tolist())
    #这些人说的话
    converContext = ann1[ann1.sender.isin(mentionNames)&(ann1.createTime > mindate)&(ann1.createTime < maxdate)]
    timediff = converContext[converContext.thread == threadid]['createTime'] - msgdate
    timediff = timediff.tolist()
    totalTimediff.extend([x.total_seconds() for x in timediff])
    
#估计这个timediff的参数
plt.hist(totalTimediff, normed = True)
#减去均值，使其为0
avg = sum(totalTimediff) / float(len(totalTimediff))
totalTimediff = [x - avg for x in totalTimediff]
st.norm.fit(totalTimediff)#loc = 0 scale = 574.8
#%%

#%%
'''
temporal context

'''
totalTimediff = []
for row in ann1.itertuples():
    msgid = row[2]
    sender = row[6]
    msgdate = row[4]
    maxdate = msgdate + datetime.timedelta(days = 1)
    mindate = msgdate - datetime.timedelta(days = 1)
    threadid = row[10]
    
    temporalContext = ann1[(ann1.createTime > mindate)&(ann1.createTime < maxdate)&(ann1.msgSvrId != msgid)]
    timediff = temporalContext[temporalContext.thread == threadid]['createTime'] - msgdate
    timediff = timediff.tolist()
    totalTimediff.extend([x.total_seconds() for x in timediff])
    
#估计这个timediff的参数
plt.hist(totalTimediff, normed = True)
#减去均值，使其为0
avg = sum(totalTimediff) / float(len(totalTimediff))
totalTimediff = [x - avg for x in totalTimediff]
st.norm.fit(totalTimediff)#loc = 0 scale = 7029.3
#%%

#%%
'''
compute the context-free freatures.
represent sentences with vectors of terms counts.
Terms:
    1.bag of words
    2.length ?
'''
#get bag of words
#split sentences
'''
#问题：是否对msg做一些过滤，字段
1.通过停止词
2.是否过滤掉@微信名这种
3.过滤掉只出现过一次的词
'''
re_filter = re.compile(u'@(?P<atname>.*?)\u2005')
ann1['msg'] = ann1.msg.str.replace(re_filter,'')#replace the @wechatid with '', this should be done when deal with predict dataset.
ann2['msg'] = ann2.msg.str.replace(re_filter,'')

import os
LTP_DATA_DIR = r'D://ltp_data//ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
#在此基础上加入分词的语句
from pyltp import Segmentor
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型

wordsLists = []

for index, msg in ann2['msg'].iteritems():
    #print(msg)
    sents = SentenceSplitter.split(msg)
    #sentsLists.append(sents)
    wordslist = []
    for sent in sents:
        #过滤句子中的标点符号
        words = segmentor.segment(sent)
        wordslist.extend(list(words))
    wordsLists.append(wordslist)
ann2['splitwords'] = wordsLists

segmentor.release()  # 释放模型,分词完毕
'''
#the filted msg may contain nothing, cause everything is filted
#remove common words
with open(r'stopwords.tab', encoding = 'utf-8') as f_stop:
     stopwords = [line.strip() for line in f_stop]
     
ann2['splitwords'] = [[word for word in wordslist if word not in stopwords] for wordslist in ann2['splitwords'].tolist()]
'''
#%%

#%%
'''
BUILD THE BAG OF WORDS
'''
from gensim import corpora, models
from six import iteritems
dictionary = corpora.Dictionary(ann2.splitwords.tolist())
#read stopwords list
with open(r'stopwords.tab', encoding = 'utf-8') as f_stop:
     stopwords = [line.strip() for line in f_stop]
#remove stop words and words appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id]
#alternative stop_ids
#dictionary.filter_n_most_frequent(50)
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]

dictionary.filter_tokens(stop_ids + once_ids)
dictionary.compactify()# remove gaps in id sequence after words that were removed

#initialize the bag of words model
corpus = [dictionary.doc2bow(text) for text in ann2['splitwords'].tolist()]
tfidf = models.TfidfModel(corpus)
corpus_tfidf =  tfidf[corpus]
#model persistency
#dictionary.save('ann2_split_words.dict')#store the dictionary for future reference
#corpus_tfidf.save('corpus_tfidf')
'''
for doc in corpus_tfidf:
    print(doc)
'''
ann2['tfidf'] = list(corpus_tfidf)
#%%

#%%
'''
BY NOW THE TFIDF MODEL IS THE CONTEXT FREE REPRESENTATION OF MESSAGES.
NOW I WILL EXPAND THIS REPRESENTATION WITH CONTEXT INFO WITH PRAMETERS TRAINED BEFORE
HOW TO EXPAND: FOR EACH MESSAGE i BUILD THE FOLLOWING FUNCTIONS
    AUTHOR CONTEXT PROB autherProb(msgi)
    CONVERSATIONAL CONTEXT PROB converProb(msgi)
    TEMPORAL CONTEXT PROB
    BOTH OF THESE FUNCTIONS RESTRICT THE TIMEWINDOW BETWEEN [DATE - 1DAY,DATE + 1DAY](to achieve this, build a df by date)
    
'''
def probMultiTfidf(probArray, tfidfList):
    '''Multiply the context tfidf with probability in the same thread.
    Args:
        probArray(Array): probability in the same thread of msg context
        tfidfList(List): the tfidfList of msg context
    Returns:
        The return tuple
    '''    
    rList = []
    for i in range(len(probArray)):
        r2List = []
        for term, freq in tfidfList[i]:
            r2List.append((term, freq*probArray[i]))
        rList.append(r2List)
    return(rList)
    
def sumTfidf(tfidfList):
    '''Sum the new computed tfidf
    Args:
        tfidfList: A list of tfidf
    Return:
        the return list: expanded tfidf dict
    '''
    rList = {}
    for tfidf in tfidfList:
        for term, freq in tfidf:
            if(term in rList):#has_key was removed in Python3
                rList[term] = rList[term] + freq
            else:
                rList[term] = freq
    return(rList)
def autherProb(row, t_scale, contextdf):
    '''Expand the context-free info of msgi with auther context.
    
    Args:
        row(Series): One raw of msgdf.itertuples() to be expanded
        t_scale(float): The scale of normal distribution trained before
        contextdf(dataframe): The dataframe of msgs between [DATE - 1DAY,DATE + 1DAY]
    Returns:
        The return vector. The expand vector repretation of msgi with info from its auther context.
    '''
    msgid = row[2]
    sender = row[6]
    msgdate = row[4]
    #maxdate = msgdate + datetime.timedelta(days = 1)
    #mindate = msgdate - datetime.timedelta(days = 1)
    #threadid = row[10] 
    #based on the defination, the autherContext dont have to be in the same thread, just the same auther.
    #FIXED
    autherContext = contextdf[(contextdf.sender == sender)&(contextdf.msgSvrId != msgid)]
    timediff = autherContext['createTime'] - msgdate
    timediff = timediff.tolist()#in seconds
    timediff = [x.total_seconds() for x in timediff]
    tfidfList = autherContext['tfidf'].tolist()
    probArray = st.norm.pdf(timediff, scale = t_scale)
    newTfidf = probMultiTfidf(probArray,tfidfList)

    return(sumTfidf(newTfidf))

def converProb(row, t_scale, contextdf, msgat):
    '''Expand the context-free info of msgi with conversational context.
    
    Args:
        row(Series): One raw of msgdf.itertuples() to be expanded
        t_scale(float): The scale of normal distribution trained before
        contextdf(dataframe): The dataframe of msgs between [DATE - 1DAY,DATE + 1DAY]
        msgat(dataframe): The dataframe of msgat between [DATE - 1DAY,DATE + 1DAY]
                        , for the reduction of repeted searching the bounded date of msgat
    Returns:
        The return vector. The expand vector repretation of msgi with info from its conversational context.    
    '''
    msgid = row[2]
    sender = row[6]
    msgdate = row[4]
    #maxdate = msgdate + datetime.timedelta(days = 1)
    #mindate = msgdate - datetime.timedelta(days = 1)
    #threadid = row[10] 
    #autherContext = contextdf[(contextdf.sender == sender)&(contextdf.msgSvrId != msgid)&(contextdf.thread == threadid)]
    #我提到了那些人
    mentionNames = msgat[msgat.msgSvrId == msgid]['member_x'].tolist()
    #我被那些人提到
    mentionNames.extend(msgat[msgat.member_x == sender]['sender'].tolist())
    #这些人说的话
    converContext = contextdf[(contextdf.sender.isin(mentionNames))]
    
    timediff = converContext['createTime'] - msgdate
    timediff = timediff.tolist()#in seconds
    timediff = [x.total_seconds() for x in timediff]
    tfidfList = converContext['tfidf'].tolist()
    probArray = st.norm.pdf(timediff, scale = t_scale)
    newTfidf = probMultiTfidf(probArray,tfidfList)

    return(sumTfidf(newTfidf))    
def tempProb(row, t_scale, contextdf):
    '''Expand the context-free info of msgi with temporal context.
    
    Args:
        row(Series): One raw of msgdf.itertuples() to be expanded
        t_scale(float): The scale of normal distribution trained before
        contextdf(dataframe): The dataframe of msgs between [DATE - 1DAY,DATE + 1DAY]

    Returns:
        The return vector. The expand vector repretation of msgi with info from its temporal context. 
    
    '''
    msgid = row[2]
    #sender = row[6]
    msgdate = row[4]

    #threadid = row[10] 
    tempContext = contextdf[(contextdf.msgSvrId != msgid)]
    timediff = tempContext['createTime'] - msgdate
    timediff = timediff.tolist()#in seconds
    timediff = [x.total_seconds() for x in timediff]
    tfidfList = tempContext['tfidf'].tolist()
    probArray = st.norm.pdf(timediff, scale = t_scale)
    newTfidf = probMultiTfidf(probArray,tfidfList)

    return(sumTfidf(newTfidf))    
def sumDicts(dict1,dict2):
    '''Sum dict1 with dict2, only 2.
    
    '''
    return({k:dict1.get(k,0) + dict2.get(k,0) for k in dict1.keys()|dict2.keys()})
def expandedMsg(contextFree, autherContext, converContext, tempContext):
    '''Sum the context-free msg tfidf with all context info.
    Args:
        contextFree(Series): The tfidf Series of the raw msg
        autherContext(List): The tfidf List of expanded auther context info.
        tempContext(List): The tfidf List of expanded temporal context info.
        converContext(List): The tfidf List of expanded conversational context info.
    Returns:
        The return vector of expanded representation.
    '''
    #TODO
    rList = []
    for i in range(len(contextFree)):
        rList.append(sumDicts(sumDicts(sumDicts(contextFree[i],autherContext[i]),converContext[i]),tempContext[i]))
    return(rList)
auther_scale = 3520
conver_scale = 574.8
temporal_scale = 7029.3
#access the date property with a .dt accessor
autherExpandList = []
converExpandList = []
tempExpandList = []
for date in ann2['createTime'].dt.date.unique():
    #FIEXED
    maxdate = datetime.datetime(date.year,date.month,date.day) + datetime.timedelta(days = twindow+1)
    mindate = datetime.datetime(date.year,date.month,date.day) - datetime.timedelta(days = twindow)
    #slice the msgat dataframe with bounded time period
    msgatdf = msgat[(msgat.createTime > mindate)&(msgat.createTime < maxdate)]
    contextdf = ann2[(ann2.createTime > mindate)&(ann2.createTime < maxdate) ]
    targetdf = ann2[ann2.createTime.dt.date == date]
    #TODO
    #check results    
    for row in targetdf.itertuples():
        autherExpandList.append(autherProb(row, auther_scale, contextdf))
        converExpandList.append(converProb(row, conver_scale,  contextdf, msgatdf))
        tempExpandList.append(tempProb(row, temporal_scale, contextdf))
        #break
#ann2['autherExpand'] = autherExpandList
ann2['tfidf'] = list(map(lambda x: dict(x), ann2['tfidf']))
tmp = expandedMsg(ann2['tfidf'].tolist(),autherExpandList,converExpandList,tempExpandList)
#%%