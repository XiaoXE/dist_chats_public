# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 14:08:41 2018

@author: Eric

run order:2

#TODO
cause MaxentClassifier.train needs the feature to be a list, we need to
0.annotate the chat log
1.combute the features of utterances in dataframe and then convert to list with a label
2.train the Maxent and then predict the whole chat log

"""
import pandas as pd
import os
os.chdir(r'D:\\0Knowledge\\Fudan\\0.20170412kidswant\\wechat\\py_wechat')


#%%
sample_msg = pd.read_pickle(r'../records/sample/sample_msg_pickle')
#%%

#%%

#%%
#create a dataframe containing the featureset
total_msg.sort_values(by = 'createTime')
total_msg[pd.isna(total_msg.createTime)].head()

