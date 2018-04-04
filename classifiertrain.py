# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:47:31 2018

@author: Eric

run order:1

"""
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
#set the time window 群组讨论的当天和前后一天
twindow = 1#这个应该用在全局的totalmsg表上，每次处理响应窗口的数据

#context info of message
#author context
    
ann1.dropna(inplace = True, subset = ['msg'])
totalTimediff = list()
for wechatid in ann1.sender.unique().tolist():
    autherContext = ann1[(ann1.sender == wechatid)&(ann1.thread > -2)]
    #totalnum = len(autherContext)
    for row in autherContext.itertuples():
        basetime = row[4]
        threadid = row[10]
        timediff = autherContext[autherContext.thread == threadid]['createTime'] - basetime
        timediff = timediff.tolist()#单位秒
        timediff = [x.total_seconds() for x in timediff]
        totalTimediff.append(timediff)
#估计这个timediff的参数
totalTimediff = [item for sublist in totalTimediff for item in sublist]
        

'''
the author context of a message m, denoted by CA(m), is the set of other messages
written by m’s author am:
'''

#%%

