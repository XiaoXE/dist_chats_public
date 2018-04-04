# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:24:09 2018

@author: Eric
"""

tmpdf = pd.DataFrame()
for room in displaynames.roomid.unique():
    members = chatroom180310[chatroom180310.chatroomname == room].memberlist.tolist()[0].split(';')
    members = pd.Series(members)
    tmp = pd.concat([displaynames[displaynames.roomid == room],members],axis = 1, ignore_index = True)
    tmpdf = pd.concat([tmpdf,tmp],ignore_index = True)