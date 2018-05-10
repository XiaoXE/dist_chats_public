# -*- coding: utf-8 -*-
"""
Created on 2018/4/28 17:37

@author: Eric

Summary: 
copy the base functions from classifiertrain.py
"""
import datetime
import scipy.stats as st
import numpy as np
import multiprocessing as mp

def sumDicts(dict1,dict2):
    """Sum dict1 with dict2, only 2.
    """
    return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in dict1.keys() | dict2.keys()}

def sumTfidf(tfidfList):
    """Sum the new computed tfidf
    Args:
        tfidfList: A list of tfidf
    Return:
        the return list: expanded tfidf dict
    """
    rList = {}
    for tfidf in tfidfList:
        rList = sumDicts(tfidf, rList)
    return rList

def probMultiTfidf(probArray, tfidfList):
    """Multiply the context tfidf with probability in the same thread.
    Args:
        probArray(Array): probability in the same thread of msg context
        tfidfList(List): the tfidfList of msg context
    Returns:
        The return tuple
    """
    return ([{k: v for k, v in zip(tfidfList[i].keys(), probArray[i] * np.array(list(tfidfList[i].values())))} for i in
             range(len(probArray))])

def ExpandDF(totalmsgdf, msgatdf, w_contextFree, w_auther, w_conver, w_temp, auther_scale, conver_scale, temporal_scale, divideN, twindow):
    """Expand the raw msg in a Dataframe with its context info.
    :param twindow: time window when expanding raw msg
    :param temporal_scale: as name shows
    :param conver_scale: as name shows
    :param auther_scale: as name shows
    :param divideN: Divide time diff with this number
    :param totalmsgdf: The target Dataframe to extend
    :param msgatdf:  The msgat Dataframe containing the @ name in each raw msg
    :param w_contextFree: weight of raw msg
    :param w_auther:  weight of auther context
    :param w_conver:  weight of conversational context
    :param w_temp:  weight of temperal context
    :return:  4 Lists, should be combined using expandedMsg function.
    """
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

    def expandedMsg(contextFree, autherContext, converContext, tempContext, w_contextFree):
        """Sum the context-free msg tfidf with all context info.
        Args:
            contextFree(List): The tfidf list of the raw msg, each element is a dict
            autherContext(List): The tfidf List of expanded auther context info.
            tempContext(List): The tfidf List of expanded temporal context info.
            converContext(List): The tfidf List of expanded conversational context info.
            w_contextFree(float): Weights for context-free
        Returns:
            The return vector of expanded representation.
        """
        return [sumDicts({k: (v * (1 - w_contextFree)) for k, v in
                          sumDicts(sumDicts(autherContext[i], converContext[i]), tempContext[i]).items()},
                         {k: (v * w_contextFree) for k, v in contextFree[i].items()}) for i in range(len(contextFree))]


    auther_expand_list, conv_expand_list, temp_expand_list, msgidlist = [],[],[],[]
    for date in totalmsgdf['createTime'].dt.date.unique():
        for talker in totalmsgdf['talker'].unique():
            maxdate = datetime.datetime(date.year, date.month, date.day) + datetime.timedelta(days=twindow + 1)
            mindate = datetime.datetime(date.year, date.month, date.day) - datetime.timedelta(days=twindow)
            # slice the msgat dataframe with bounded time period
            msgat = msgatdf[(msgatdf.createTime > mindate) & (msgatdf.createTime < maxdate) & (msgatdf.talker == talker)]

            # 还需要限制在同一个聊天室中
            contextdf = totalmsgdf[(totalmsgdf.createTime > mindate) & (totalmsgdf.createTime < maxdate) & (totalmsgdf.talker == talker)]
            targetdf = totalmsgdf[(totalmsgdf.createTime.dt.date == date) & (totalmsgdf.talker == talker)]
            cnt = 0
            for row in targetdf.itertuples():
                print(date, cnt)
                cnt +=1
                msgidlist.append(row[1])
                auther_expand_list.append(autherProb(row, auther_scale, contextdf, w_auther))
                conv_expand_list.append(converProb(row, conver_scale, contextdf, msgat, w_conver))
                temp_expand_list.append(tempProb(row, temporal_scale, contextdf, w_temp))
            '''
            [[autherProb(row, auther_scale, contextdf, w_auther),converProb(row, conver_scale, contextdf, msgat, w_conver),tempProb(row, temporal_scale, contextdf, w_temp)] for row in targetdf.itertuples()]
            '''
    return msgidlist,auther_expand_list,conv_expand_list,temp_expand_list
    #下面的return的结果是不对的，因为auther_expand_list中都是按照日期和聊天房组织的，但totalmsgdf却不是，所以会造成乱序
    #return expandedMsg(totalmsgdf['tfidf'].tolist(), auther_expand_list, conv_expand_list, temp_expand_list,w_contextFree)
