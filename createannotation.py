# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:49:22 2018

@author: Eric

run order 1.0
"""

import pandas as pd
import os
os.chdir(r'D:\\0Knowledge\\Fudan\\0.20170412kidswant\\wechat\\py_wechat')


#%%
sample_msg = pd.read_pickle(r'../records/sample/sample_msg_pickle')
#7354409773@chatroom     4229
#7445858570@chatroom     4332
sample1 = sample_msg[(sample_msg.talker == '7354409773@chatroom')&(sample_msg.createTime.dt.month == 6)]
sample1.groupby(sample1.createTime.dt.day).size()
anno1 = sample1[sample1.createTime.dt.day.isin([12,13,14,15])]#730
sample2 = sample_msg[(sample_msg.talker == '7445858570@chatroom')&(sample_msg.createTime.dt.month == 6)]
sample2.groupby(sample2.createTime.dt.day).size()
anno2 = sample2[sample2.createTime.dt.day.isin([16,17,18,19])]#773

anno2.drop(columns = ['isSend','status'],inplace = True)
anno1.drop(columns = ['isSend','status'],inplace = True)
#to csv and transfer the code using notepad++
anno1.to_csv(r'../records/annotation/anno1.csv')
anno2.to_csv(r'../records/annotation/anno2.csv')
#标注添加两列，thread和appthread
#标注的时候，应用信息用-2表示，系统提示用-1表示，其他整数表示
