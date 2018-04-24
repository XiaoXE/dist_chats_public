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
import math
import timeit
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
    timediff = [x.total_seconds()/60 for x in timediff]
    totalTimediff.extend(timediff)

#估计这个timediff的参数
plt.hist(totalTimediff, normed = True)
plt.show()
st.norm.fit(totalTimediff)#loc = 0 scale = 58.7
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
    totalTimediff.extend([x.total_seconds()/60 for x in timediff])
    
#估计这个timediff的参数
plt.hist(totalTimediff, normed = True)
plt.show()
#减去均值，使其为0
avg = sum(totalTimediff) / float(len(totalTimediff))
totalTimediff = [x - avg for x in totalTimediff]
st.norm.fit(totalTimediff)#loc = 0 scale = 9.6
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
    totalTimediff.extend([x.total_seconds()/60 for x in timediff])
    
#估计这个timediff的参数
plt.hist(totalTimediff, normed = True)
#减去均值，使其为0
avg = sum(totalTimediff) / float(len(totalTimediff))
totalTimediff = [x - avg for x in totalTimediff]
st.norm.fit(totalTimediff)#loc = 0 scale = 117.2
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
ann2['tfidf'] = list(map(lambda x: dict(x), ann2['tfidf']))# to dict
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
    # alternative
    return ([{k: v for k, v in zip(tfidfList[i].keys(), probArray[i] * np.array(list(tfidfList[i].values())))} for i in
             range(len(probArray))])



def sumTfidf(tfidfList):
    '''Sum the new computed tfidf
    Args:
        tfidfList: A list of tfidf
    Return:
        the return list: expanded tfidf dict
    '''
    rList = {}
    for tfidf in tfidfList:
        rList = sumDicts(tfidf, rList)
    return(rList)
def autherProb(row, t_scale, contextdf, w_auther):
    '''Expand the context-free info of msgi with auther context.
    
    Args:
        row(Series): One raw of msgdf.itertuples() to be expanded
        t_scale(float): The scale of normal distribution trained before
        contextdf(dataframe): The dataframe of msgs between [DATE - 1DAY,DATE + 1DAY]
        w_auther(float): Weight for auther context.
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
    timediff = [x.total_seconds()/3600 for x in timediff]
    tfidfList = autherContext['tfidf'].tolist()
    probArray = st.norm.pdf(timediff, scale = t_scale)
    newTfidf = probMultiTfidf(probArray,tfidfList)

    #return(sumTfidf(newTfidf))
    return({k:(w_auther*v)  for k,v in sumTfidf(newTfidf).items()})

def converProb(row, t_scale, contextdf, msgat, w_conver):
    '''Expand the context-free info of msgi with conversational context.
    
    Args:
        row(Series): One raw of msgdf.itertuples() to be expanded
        t_scale(float): The scale of normal distribution trained before
        contextdf(dataframe): The dataframe of msgs between [DATE - 1DAY,DATE + 1DAY]
        msgat(dataframe): The dataframe of msgat between [DATE - 1DAY,DATE + 1DAY]
                        , for the reduction of repeted searching the bounded date of msgat
        w_conver(float): weight for conversational context
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
    timediff = [x.total_seconds()/3600 for x in timediff]
    tfidfList = converContext['tfidf'].tolist()
    probArray = st.norm.pdf(timediff, scale = t_scale)
    newTfidf = probMultiTfidf(probArray,tfidfList)

    return({k:(v*w_conver) for k,v in sumTfidf(newTfidf).items()})    
def tempProb(row, t_scale, contextdf, w_temp):
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
    timediff = [x.total_seconds()/3600 for x in timediff]
    tfidfList = tempContext['tfidf'].tolist()
    probArray = st.norm.pdf(timediff, scale = t_scale)
    newTfidf = probMultiTfidf(probArray,tfidfList)

    return({k:(v*w_temp)for k,v in sumTfidf(newTfidf).items()})
def sumDicts(dict1,dict2):
    '''Sum dict1 with dict2, only 2.
    '''
    return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in dict1.keys() | dict2.keys()}
def expandedMsg(contextFree, autherContext, converContext, tempContext, w_contextFree):
    '''Sum the context-free msg tfidf with all context info.
    Args:
        contextFree(List): The tfidf list of the raw msg, each element is a dict
        autherContext(List): The tfidf List of expanded auther context info.
        tempContext(List): The tfidf List of expanded temporal context info.
        converContext(List): The tfidf List of expanded conversational context info.
        w_contextFree(float): Weights for context-free
    Returns:
        The return vector of expanded representation.
    '''
    '''
    rList = []
    for i in range(len(contextFree)):
        #rList.append(sumDicts(sumDicts(sumDicts(contextFree[i],autherContext[i]),converContext[i]),tempContext[i]))
        contextPart ={k:(v*(1-w_contextFree))for k,v in sumDicts(sumDicts(autherContext[i],converContext[i]),tempContext[i]).items()}
        contextFreePart = {k:(v*w_contextFree) for k,v in contextFree[i].items()}
        rList.append(sumDicts(contextFreePart, contextPart))# can be optimized
        
    return(rList)
    '''

    #alternative1 : to dict comprehension
    return [sumDicts({k:(v*(1-w_contextFree))for k,v in sumDicts(sumDicts(autherContext[i],converContext[i]),tempContext[i]).items()}, {k:(v*w_contextFree) for k,v in contextFree[i].items()}) for i in range(len(contextFree))]
    #alternative2 : to numpy array? no!

auther_scale = 58.7
conver_scale = 9.6
temporal_scale = 117
w_contextFree = 0.45

w_auther = 0.3
w_conver = 0.6
w_temp = 0.1
#weights = {'w_contextFree':w_contextFree,'w_context':w_context,'w_auther':w_auther,'w_conver':w_conver,'w_temp':w_temp}
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
    for row in targetdf.itertuples():
        autherExpandList.append(autherProb(row, auther_scale, contextdf, w_auther))
        converExpandList.append(converProb(row, conver_scale, contextdf, msgatdf, w_conver))
        tempExpandList.append(tempProb(row, temporal_scale, contextdf, w_temp))
        #break
ann2['extended'] = expandedMsg(ann2['tfidf'].tolist(),autherExpandList,converExpandList,tempExpandList, w_contextFree)
#%%

#%%
'''Compute the similarity between a given msg and existing threads
Single pass clustering is performed.(reference about single pass tech: 
    http://facweb.cs.depaul.edu/mobasher/classes/csc575/assignments/single-pass.html)
1.Treat the first msg as a single-message cluster T.
2.for each remaining msg m compute for All the existing threads.
    sim(m, T)= max sim(m, mi) mi belongs to All the existing threads.
    sim(m, mi) = cosin similarity between these two msgs.
'''
def dictNorm(dict1):
    '''Compute the norm of a dictionary just like a vector
    Args:
        dict1(dict): the square of sum of each value in this dict.
    Returns:
        The return float.
    '''    
    return(sum([v*v for v in dict1.values()]))
def dotProduct(dict1,dict2):
    '''Compute the dot product of two dicts, like vectors
    Args:
        dict1, dict2(dicts): two tfidf dicts.
    Returns:
        The return float.
    '''
    return sum([dict1[k] * dict2[k] for k in dict1.keys() & dict2.keys()])

def similarity(msgdf, targetmsgid, msgdate, threadDict, threshold):
    '''Pairwise similarity function.
    
    1. Turn the date into date counts.
    2. The composation of Theadid:
        Part1: Date count * 10e4
        Part2: increamental thread id counts.(suppose this count will not exceed 10e3)
        Theadid = Part1 + Part2
    3. Only compute in the range of [Date - 1, Date]
    Args:
        msgdf(dataframe): The dataframe of msg log
        targetmsgid(String): The target unique msg id to be compared.
        msgdate(int): The date count from the beginning date.
        threadDict(dict): key= thread id, value = List of msgid.
        threshold(float): if the max similarity is under threshold, start a new cluster containing only this msg
    Returns:
        The updated threadDict.
    '''
    if len(threadDict) == 0:
        threadid = 1 + 10000*msgdate
        threadDict[threadid] = [targetmsgid]
    else:
        max_similarity_thread = 0
        max_similarity = 0
        #find the max similarity and the corresponding thread over all threads.
        for thread, msgids in threadDict.items():
            #Notice, the threadDict may be NULL!
            threaddate = thread // 10000
            if threaddate < msgdate - 1: continue
            for msgid in msgids:
                targetmsg = msgdf[msgdf.msgSvrId == targetmsgid]['extended'].item()#get the exact dict rather than dict and object type
                comparedmsg = msgdf[msgdf.msgSvrId == msgid]['extended'].item()
                cosine = dotProduct(targetmsg,comparedmsg)/math.sqrt(dictNorm(targetmsg)*dictNorm(comparedmsg))
                if (cosine > max_similarity):
                    max_similarity = cosine
                    max_similarity_thread = thread
                     
        if max_similarity > threshold:
            #print(max_similarity)
            threadDict[max_similarity_thread].append(targetmsgid)
        else:
            #CAUTION maybe wrong
            #create a new thread
            if threaddate == msgdate:
                threadDict[max(threadDict.keys())%10000 + 1 + msgdate*10000] = [targetmsgid]
            if threaddate < msgdate:
                #if this a new date and new thread, then reset the thread id to 1.
                threadDict[1 + msgdate*10000] = [targetmsgid]
    return threadDict

threadDict = {}       
begin_date = min(ann2['createTime'])
threshold = 0.7
# Use count to control how many msgs to distengle thread.
#count = 1
for row in ann2.itertuples():
    msgid = row[2]
    msgdate = row[4]
    msgdate = (msgdate - begin_date).days + 1
    threadDict = similarity(ann2, msgid, msgdate, threadDict, threshold)
    #count += 1
    #if(count >50): break
#for k,v in threadDict.items():
#    print(k, len(v))

# compare the performance between for loop map() and list comprehension
# map is not suitable for this iteration.
# and list comprehension is not suitable too.
#%%
    
#%%
'''Choose the F value to be the object and try to train the optimal model.
1. Build the F value.
2. Adjust the parameters to maximize F value.
''' 
def recall(ti,tj):
    """The recall between the real thread i and the detected thread j.

    Recall(i,j) = nij / ni
    where nij is the number of msgs of the real thread i in the detected thread j.
    ni is the number of msgs in the real thread i.

    Args:
        ti(int): The real thread number, also the key in the realThreadDict.
        tj(int): The detected thread number, also the key in the threadDict.
    Return:
        The return float number.
    """
    realMsg = realThreadDict[ti]
    detectMsg = threadDict[tj]
    nij = len([real for real in realMsg if real in detectMsg])
    ni = len(realMsg)
    # check the result with jupyter console
    # good result
    return nij / ni
def precision(ti,tj):
    """Precision(i,j) = nij / nj
    where nj is the number of msgs in the detected thread j.

    Args:
        ti(int): The real thread number, also the key in the realThreadDict.
        tj(int): The detected thread number, also the key in the threadDict.
    Return:
        The return float number.
    """
    realMsg = realThreadDict[ti]
    detectMsg = threadDict[tj]
    nij = len([real for real in realMsg if real in detectMsg])
    nj = len(detectMsg)
    # check the result with jupyter console
    # good result
    return nij / nj

def pairf(ti,tj):
    """ F(i,j) = 2*Precision*Recall /(Precision + Recall)
    is the F measure of detected thread j and the real thread i.
    """
    prevalue = precision(ti,tj)
    revalue = recall(ti,tj)
    #print(prevalue, revalue)
    if (prevalue == 0)|(revalue == 0):
        return 0
    else:
        return 2 * prevalue * revalue / (prevalue + revalue)

def fvalue(realThreadDict, threadDict):
    """The whole F measure of the detection result in a stream
is defined as a weighted sum over all threads as follow.
    Args:
        realThreadDict(dict): The dict of ground true threads of msgs. Key is thread id and value is msgid list.
        threadDict(dict): The dict of detected thread.
    """
    max_pairf, wholef = 0, 0
    len_msg = len(ann2)
    for realThread in realThreadDict:
        for detectThread in threadDict:
            value_pairf = pairf(realThread, detectThread)
            if value_pairf > max_pairf: max_pairf = value_pairf
        wholef = wholef + len(realThreadDict[realThread])*value_pairf

    return wholef/len_msg

realThreadDict = {}
for thread in ann2['thread'].unique():
    msgs = ann2[ann2['thread'] == thread]['msgSvrId'].tolist()
    realThreadDict[thread] = msgs

fvalue(realThreadDict, threadDict)
plt.bar(realThreadDict.keys(),[len(value) for value in realThreadDict.values()] )
plt.show()
plt.bar(range(len(threadDict)),[len(value) for value in threadDict.values()] )
plt.show()
#%% Tuning Function
def tuning(w_contextFree=0.45, w_auther=0.3, w_conver=0.6, w_temp=0.1, threshold=0.7):
    """Tuning parameters.
    1. Compute the extended msg tfidf with all the parameters.
    2. Cluster msgs into threads.
    3. Compute the F value.

    Args:
        w_*, threshold(float): all the weights and threshold to be tuning.

    """
    # access the date property with a .dt accessor
    autherExpandList = []
    converExpandList = []
    tempExpandList = []
    for date in ann2['createTime'].dt.date.unique():
        # FIEXED
        maxdate = datetime.datetime(date.year, date.month, date.day) + datetime.timedelta(days=twindow + 1)
        mindate = datetime.datetime(date.year, date.month, date.day) - datetime.timedelta(days=twindow)
        # slice the msgat dataframe with bounded time period
        msgatdf = msgat[(msgat.createTime > mindate) & (msgat.createTime < maxdate)]
        contextdf = ann2[(ann2.createTime > mindate) & (ann2.createTime < maxdate)]
        targetdf = ann2[ann2.createTime.dt.date == date]
        for row in targetdf.itertuples():
            autherExpandList.append(autherProb(row, auther_scale, contextdf, w_auther))
            converExpandList.append(converProb(row, conver_scale, contextdf, msgatdf, w_conver))
            tempExpandList.append(tempProb(row, temporal_scale, contextdf, w_temp))

    # ann2['tfidf'] = list(map(lambda x: dict(x), ann2['tfidf']))
    # ann2['extended'] = expandedMsg(ann2['tfidf'].tolist(),autherExpandList,converExpandList,tempExpandList, w_contextFree)
    # extendedtfidf is a list of dicts.
    extendedtfidf = expandedMsg(ann2['tfidf'].tolist(), autherExpandList, converExpandList, tempExpandList,
                                w_contextFree)

    # Then compute the similarity and get the thread.
    threadDict = {}
    begin_date = min(ann2['createTime'])
    for row in ann2.itertuples():
        msgid = row[2]
        msgdate = row[4]
        msgdate = (msgdate - begin_date).days + 1
        threadDict = similarity(ann2, msgid, msgdate, threadDict, threshold)

    # Then compute the F value.
    return fvalue(realThreadDict, threadDict)

#TODO
#BASED ON DIFF_REAL_DETECTED_THREAD, FIX SIMILARITY FUNCTION, FIND OUT THE REASON WHY SO MANY TINY THREADS.