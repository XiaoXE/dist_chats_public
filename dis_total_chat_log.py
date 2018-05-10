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
#import re
import pandas as pd
#from pyltp import SentenceSplitter
import time
import datetime
import scipy.stats as st
import numpy as np
import multiprocessing as mp
import os

#%% Expand msg
# prepare data
msgat = pd.read_pickle(r'../records/sample/msgat_dataframe')
total_msg = pd.read_pickle(r'../records/sample/sample_msg_tfidf_pickle')
# test a subsample of total_msg
total_msg = total_msg.iloc[:1000]
# %% initilize paremeters
twindow = 0.5  # 这个应该用在全局的totalmsg表上，每次处理相应窗口的数据
divideN = 1500  # 对timediff进行缩放

threshold = 0.6

w_contextfree = 0.14
w_auther = 0.3
w_conver = 0.6
w_temp = 0.1

auther_scale = 2.347
conver_scale = 0.383
temporal_scale = 4.686

#print some info
def info(title):
    print(title, 'starts at {}'.format(time.time()))
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
#import classifier_base_functions as classbase
from classifier_base_functions import probMultiTfidf,sumTfidf

def mExpand(msg_date_talker):
    info('function mExpand')
    msgdate = msg_date_talker[0]
    talker = msg_date_talker[1]
    """Worker function. 并行处理的每个单位为某date的total_msg

    :return: List containing expanded msg dicts
    """
    #print('start processing')
    def autherProb(row, t_scale, contextdf, w_auther):
        """Expand the context-free info of msgi with auther context.

        Args:
            row(Series): One raw of msgdf.itertuples() to be expanded
            t_scale(float): The scale of normal distribution trained before
            contextdf(dataframe): The dataframe of msgs between [DATE - 1DAY,DATE + 1DAY]
            w_auther(float): Weight for auther context.
        Returns:
            The return vector. The expand vector repretation of msgi with info from its auther context.
        """
        msgid = row[1]
        sender = row[4]
        msgdate = row[2]

        autherContext = contextdf[(contextdf.sender == sender) & (contextdf.msgSvrId != msgid)]
        timediff = autherContext['createTime'] - msgdate
        timediff = timediff.tolist()  # in seconds
        timediff = [x.total_seconds() / divideN for x in timediff]
        tfidfList = autherContext['tfidf'].tolist()
        probArray = st.norm.pdf(timediff, scale=t_scale)
        newTfidf = probMultiTfidf(probArray, tfidfList)

        return {k: (w_auther * v) for k, v in sumTfidf(newTfidf).items()}

    def converProb(row, t_scale, contextdf, msgat, w_conver):
        """Expand the context-free info of msgi with conversational context.

        Args:
            row(Series): One raw of msgdf.itertuples() to be expanded
            t_scale(float): The scale of normal distribution trained before
            contextdf(dataframe): The dataframe of msgs between [DATE - 1DAY,DATE + 1DAY]
            msgat(dataframe): The dataframe of msgat between [DATE - 1DAY,DATE + 1DAY]
                            , for the reduction of repeted searching the bounded date of msgat
            w_conver(float): weight for conversational context
        Returns:
            The return vector. The expand vector repretation of msgi with info from its conversational context.
        """
        msgid = row[1]
        sender = row[4]
        msgdate = row[2]
        # maxdate = msgdate + datetime.timedelta(days = 1)
        # mindate = msgdate - datetime.timedelta(days = 1)
        # threadid = row[10]
        # autherContext = contextdf[(contextdf.sender == sender)&(contextdf.msgSvrId != msgid)&(contextdf.thread == threadid)]
        # 我提到了那些人
        mentionNames = msgat[msgat.msgSvrId == msgid]['member_x'].tolist()
        # 我被那些人提到
        mentionNames.extend(msgat[msgat.member_x == sender]['sender'].tolist())
        # 这些人说的话
        converContext = contextdf[(contextdf.sender.isin(mentionNames))]

        timediff = converContext['createTime'] - msgdate
        timediff = timediff.tolist()  # in seconds
        timediff = [x.total_seconds() / divideN for x in timediff]
        tfidfList = converContext['tfidf'].tolist()
        probArray = st.norm.pdf(timediff, scale=t_scale)
        newTfidf = probMultiTfidf(probArray, tfidfList)

        return {k: (v * w_conver) for k, v in sumTfidf(newTfidf).items()}

    def tempProb(row, t_scale, contextdf, w_temp):
        """Expand the context-free info of msgi with temporal context.

        Args:
            row(Series): One raw of msgdf.itertuples() to be expanded
            t_scale(float): The scale of normal distribution trained before
            contextdf(dataframe): The dataframe of msgs between [DATE - 1DAY,DATE + 1DAY]

        Returns:
            The return vector. The expand vector repretation of msgi with info from its temporal context.

        """
        msgid = row[1]
        msgdate = row[2]

        tempContext = contextdf[(contextdf.msgSvrId != msgid)]
        timediff = tempContext['createTime'] - msgdate
        timediff = timediff.tolist()  # in seconds
        timediff = [x.total_seconds() / divideN for x in timediff]
        tfidfList = tempContext['tfidf'].tolist()
        probArray = st.norm.pdf(timediff, scale=t_scale)
        newTfidf = probMultiTfidf(probArray, tfidfList)

        return {k: (v * w_temp) for k, v in sumTfidf(newTfidf).items()}


    maxdate = datetime.datetime(msgdate.year, msgdate.month, msgdate.day) + datetime.timedelta(days=twindow + 1)
    mindate = datetime.datetime(msgdate.year, msgdate.month, msgdate.day) - datetime.timedelta(days=twindow)
    # slice the msgat dataframe with bounded time period
    msgatSlice = msgat[(msgat.createTime > mindate) & (msgat.createTime < maxdate)]
    #还需要限制在同一个聊天室中
    contextdf = total_msg[(total_msg.createTime > mindate) & (total_msg.createTime < maxdate) & (total_msg.talker == talker)]
    targetdf = total_msg[(total_msg.createTime.dt.date == msgdate) & (total_msg.talker == talker)]
    # 下面这一句可以优化
    allContext = [[autherProb(row, auther_scale, contextdf, w_auther),converProb(row, conver_scale, contextdf, msgatSlice, w_conver),tempProb(row, temporal_scale, contextdf, w_temp)] for row in targetdf.itertuples()]
    auther_expand_list = [i[0] for i in allContext]
    conv_expand_list = [i[1] for i in allContext]
    temp_expand_list = [i[2] for i in allContext]
    msgidlist = targetdf.msgSvrId.tolist()
    return [msgidlist,auther_expand_list,conv_expand_list,temp_expand_list]
    #下面的返回语句没有明确到底msgid是多少
    #return expandedMsg(targetdf['tfidf'].tolist(), auther_expand_list, conv_expand_list, temp_expand_list,w_contextFree)

'''
The multiprocessing module also introduces APIs which do not have analogs in the threading module.
 A prime example of this is the Pool object which offers a convenient means of parallelizing the execution of a function across multiple input values, 
 distributing the input data across processes (data parallelism).
'''
def multiproc():
    """

    :param totalmsgdf:
    :param msgatdf:
    :param w_contextFree:
    :param w_auther:
    :param w_conver:
    :param w_temp:
    :param auther_scale:
    :param conver_scale:
    :param temporal_scale:
    :param divideN:
    :param twindow:
    :return: return list of msgid, auther_context_list, conver_context_list, temp_context_list
    """
    with mp.Pool(processes=4) as p:
        #pool.apply can maintain an ordered list
        #expanded = [p.apply(mExpand,args= (totalmsgdf, msgdate,talker, msgatdf, w_contextFree, w_auther, w_conver, w_temp, auther_scale, conver_scale, temporal_scale, divideN, twindow))
        #           for msgdate in totalmsgdf['createTime'].dt.date.unique() for talker in totalmsgdf['talker'].unique()]
        expanded = p.map(mExpand,[[msgdate, talker] for msgdate in total_msg['createTime'].dt.date.unique() for talker in total_msg['talker'].unique()])
    return expanded

if __name__ == '__main__':

    start_time = time.time()
    info('main line')
    #multiprocessing 133s 1275s(5000)
    #m_result = classbase.multiproc(total_msg,msgat,w_contextfree,w_auther,w_contextfree,w_temp, auther_scale, conver_scale, temporal_scale, divideN, twindow)
    #m_result = classbase.multiproc_async(total_msg, msgat, w_contextfree, w_auther, w_contextfree, w_temp, auther_scale,conver_scale, temporal_scale, divideN, twindow)
    #output = [p.get() for p in m_result]
    #print(output)

    #multiproc using map function
    m_result  = multiproc()
    '''
    #####Test mExpand Function 132s
    result = []
    for msgdate in total_msg['createTime'].dt.date.unique():
        for talker in total_msg['talker'].unique():
            tmp = classbase.mExpand(total_msg,msgdate,talker,msgat,w_contextfree,w_auther,w_contextfree,w_temp, auther_scale, conver_scale, temporal_scale, divideN, twindow)
            result.append(tmp)
    #####
    '''

    #147s 1330s(5000)
    #msgidlist,auther_list, conver_list, temp_list = classbase.ExpandDF(total_msg,msgat,w_contextfree,w_auther,w_contextfree,w_temp, auther_scale, conver_scale, temporal_scale, divideN, twindow)

    print('--- {} seconds ---'.format(time.time()-start_time))
#total_msg.to_pickle(r'../records/sample/sample_msg_tfidf_expanded_pickle_woopt')

#rresult = pd.read_pickle(r'../records/sample/sample_msg_tfidf_expanded_pickle_woopt')

#TODO
#应该可以使用并行处理
#测试下性能，比如先跑1k条数据,126s
#将程序做些优化
#做个类似运行进度条之类的东西

#原始的未经优化的程序是真的慢
# Read 17.2.3. Programming guidelines
# test multiprocess map