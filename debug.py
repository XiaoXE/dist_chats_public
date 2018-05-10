import pandas as pd
import time
import datetime
import scipy.stats as st
import os
from classifier_base_functions import probMultiTfidf,sumTfidf
import pickle

# prepare data
total_msg = pd.read_pickle(r'../records/sample/sample_msg_tfidf_pickle')
# drop useless columns
total_msg.drop(columns=['type', 'status', 'isSend', 'splitwords','msg'], inplace=True)
# test a subsample of total_msg
total_msg.sort_values('createTime',inplace= True)
total_msg['intTime'] = total_msg.createTime.diff().dt.total_seconds().fillna(0).cumsum()
total_msg.sort_values(['talker','createTime'],inplace= True,ascending=True)
total_msg.index = pd.RangeIndex(len(total_msg.index))
total_msg = total_msg.iloc[:1000]
msgat = pd.read_pickle(r'../records/sample/msgat_dataframe')

def autherProb(row, t_scale, contextdf, w_auther):
    msgrowid = row.name
    sender = row[3]
    msgdate = row[7]#intTime

    autherContext = contextdf[(contextdf.sender == sender) & (contextdf.index != msgrowid)]
    timediff = autherContext['intTime'].values - msgdate# to ndarray
    timediff = timediff / divideN
    tfidfList = autherContext['tfidf'].tolist()
    probArray = st.norm.pdf(timediff, scale=t_scale)
    newTfidf = probMultiTfidf(probArray, tfidfList)
    result = {k: (w_auther * v) for k, v in sumTfidf(newTfidf).items()}
    #write to a file
    with open(r'../records/sample/auther_expand_test','a') as f:
        f.write("{0},{1}\n".format(msgrowid,result))   

def converProb(row, t_scale, contextdf, msgat, w_conver):
    msgid = row[1]
    sender = row[3]
    msgdate = row[7]#intTime
    # 我提到了那些人
    mentionNames = msgat[msgat.msgSvrId == msgid]['member_x'].tolist()
    # 我被那些人提到
    mentionNames.extend(msgat[msgat.member_x == sender]['sender'].tolist())
    # 这些人说的话
    converContext = contextdf[(contextdf.sender.isin(mentionNames))]

    timediff = converContext['intTime'].values - msgdate
    timediff = timediff / divideN
    tfidfList = converContext['tfidf'].tolist()
    probArray = st.norm.pdf(timediff, scale=t_scale)
    newTfidf = probMultiTfidf(probArray, tfidfList)
    result = {k: (v * w_conver) for k, v in sumTfidf(newTfidf).items()}
    #write to a file
    with open(r'../records/sample/conver_expand_test','a') as f:
        f.write("{0},{1}\n".format(msgid,result))       
def mExpandAutherConver(msgdate,talker):
    maxdate = datetime.datetime(msgdate.year, msgdate.month, msgdate.day) + datetime.timedelta(hours=24 + twindow)
    mindate = datetime.datetime(msgdate.year, msgdate.month, msgdate.day) - datetime.timedelta(hours=twindow)
    # slice the msgat dataframe with bounded time period and the same talker
    msgatSlice = msgat[(msgat.createTime > mindate) & (msgat.createTime < maxdate)&(msgat.talker == talker)]
    #还需要限制在同一个聊天室中
    contextdf = total_msg[(total_msg.createTime > mindate) & (total_msg.createTime < maxdate) & (total_msg.talker == talker)]
    targetdf = total_msg[(total_msg.createTime.dt.date == msgdate) & (total_msg.talker == talker)]
    #targetdf.apply(lambda row: [autherProb(row, auther_scale, contextdf, w_auther),converProb(row, conver_scale, contextdf, msgat, w_conver)],axis=1)
    targetdf.apply(lambda row:converProb(row, conver_scale, contextdf, msgat, w_conver), axis=1)

twindow = 3  # 这个应该用在全局的totalmsg表上，每次处理相应窗口的数据
divideN = 1500  # 对timediff进行缩放

threshold = 0.6

w_contextfree = 0.14
w_auther = 0.3
w_conver = 0.6
w_temp = 0.1

auther_scale = 2.347
conver_scale = 0.383
temporal_scale = 4.686
#%%
[mExpandAutherConver(msgdate,talker) for msgdate in total_msg['createTime'].dt.date.unique() for talker in total_msg['talker'].unique()]