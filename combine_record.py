# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a combined wechat record analysis script file.

run order 0.0
output: a combined wechat group chattting log

"""
#%% ddd
import numpy as np
import pandas as pd
import os
import re
import chardet
import datetime
os.chdir(r'D:\\0Knowledge\\Fudan\\0.20170412kidswant\\wechat\\py_wechat')
#%%

#%%
"""
how to get wechat records:
https://www.zhihu.com/question/19924224

tables:
message 
  send_msg凡是含有appid的，到appinfo表中进行匹配，例如appid=\"\"wx992c420c8d64dc18\"\"
          凡是含有title的，<title>少儿手足口病险（秒杀专享）</title>，也取出来
  type :1常规文本3应用消息49其他应用分享的信息，可以通过<title>获取47表情10000系统提示信息，加入群聊，撤回消息等
    34图片 43图片436207665红包 268435505 应用信息 42 估计是名片 48位置共享 1048625不明，去匹配Appmessage通过msgid
    452984881 去匹配appmessage 16777265匹配appmessage 64语音聊天的系统提示 10000进群退群提示 -1879048186去匹配
    
    基本上就是除了1以外，其他的从appmessage进行匹配
rcontact 联系人，个人的表
chatroom0830 群聊的基本信息
AppMessage 各种应用的提示信息，包括微信自己的红包和第三方的应用分享
EmojiInforDesc 表情描述
appinfor0830涉及到的应用的描述

大概就是这些会比较有用
"""

'''
首先是将微信的聊天记录合并起来
并对数据做一些初步的处理
'''
#%%
#%%
'''
#get group display name from the html source code
#parse the code with re and extract the displayname
#这里得到的是全部的群昵称
'''
#match chatroom name in Chinese with chatroom id .
match_chatroom = pd.read_csv(r'../records/match_chatroom.csv', encoding = 'gb2312')
basepath = u'../records/nicknames/'
'''
input : the directory of html source codes
output : a df with group display names
'''
def parseDisplayname(dir):
    displaydf = pd.DataFrame(columns = ['roomname','displayname'])    
    for file in os.listdir(dir):
        chatroomname = file
        with open(basepath + file,'r', encoding = 'utf-8') as f:
            htmlfile = f.read()
            
            #re.complie 将用到的正则表达式预先编码，提高速度，模式re.S，实现跨行匹配
            #注意正则表达式里里面的.和()都需要进行转义
            re_dis = re.compile(r'getDisplayName\(currentContact\.UserName\)">(.*?)</p>',re.S)
            nicknames = re_dis.findall(htmlfile)
            
            tmpdf = pd.DataFrame({
                    'roomname':chatroomname,
                    'displayname':nicknames
                    })
            displaydf = displaydf.append(tmpdf,ignore_index = True)
    return displaydf
#这里得到的昵称是群昵称，也就是聊天的时候显示的名字
#而且，这里的个数只会比chatroom的行数多吧。。。
#这里假设没有人退群
displaynames = parseDisplayname(basepath)#16356
#先假设没有人退群，做merge处理，比对通过roomdata得到的displayname
#drop 掉两个群，武进万达孕妈群和辣妈帮
displaynames.drop_duplicates(subset = ['displayname','roomid'], inplace = True)
displaynames = displaynames.merge(match_chatroom, left_on = 'roomname', right_on = 'name', how = 'inner')#16356

#%%
#%%
'''
#read
#第一个备份从5月到8月
'''
message0830 = pd.read_csv(r'../records/170830/message.csv')
chatroom0830 = pd.read_csv(r'../records/170830/chatroom.csv')
appmsg0830 = pd.read_csv(r'../records/170830/AppMessage.csv')
appinfor0830 = pd.read_csv(r'../records/170830/appinfor.csv')
#%%

#filter chatroom0830
#%%
nokidswantgroup = ["7262310752@chatroom","6510569027@chatroom"]#非孩子王的群
chatroom0830 = chatroom0830[~chatroom0830.chatroomname.isin(nokidswantgroup)]
remove_room = message0830.talker[message0830.content.str.contains(r'"移出群聊').fillna(False)]#被5个群移除群聊
chatroom0830 = chatroom0830[~chatroom0830.chatroomname.isin(remove_room)]#现在有79个群
#%%

#%%
def match_nickname(chatroomdf):
    """
    to get nicknames of members in a group, but the question is when people chat in group, the name displayed in
    group chatting is not always the nickname but sometimes the group displayname if they set this name, which can split from the roomdata column.
    But! this function is still usable for those chatroom without fault.

    In conclusion, this function get nicknames in the right chatroom
    """
    matched_df = pd.DataFrame(columns = ['chatroomname','member','nickname'])
    for row in chatroomdf.itertuples():
        chatroomname = row[1]
        memberset = row[3]
        displaynameset = row[4]
        memberlist = memberset.split(';')
        displaynamelist = displaynameset.split('、')
        if len(memberlist) == len(displaynamelist):
            for i in range(len(memberlist)):
                tmpdf = pd.DataFrame({
                        'chatroomname':chatroomname,
                        'member':memberlist[i],
                        'nickname':displaynamelist[i]                    
                        },index = [0])
                matched_df = matched_df.append(tmpdf,ignore_index=True)
    return matched_df
    
def match_groupdisplayname(chatroomdf,nicknamedf):
    """
    to match the member wechat id and group display name in roomdata, which can
    not be derived from other columns in chatroom dataframe
    """
    matched_df = pd.DataFrame(columns = ['chatroomname','member','displayname'])
    for row in chatroomdf.itertuples():
        chatroomname = row[1]        
        totalset = row[8]
        #splitby \n,but not DC2\n
        tmp1 = re.split('(?<!\x12)\n',totalset)
        #split by ASCII control codes
        tmp2 = [re.split('[\x00-\x1F]',x) for x in tmp1]
        wechatid = []
        displayname = []
        tmp3 = []
        for i in tmp2:
            if len(i) <2:
                tmp2.remove(i)
            else:
                while '' in i:
                    i.remove('')
                tmp3.append(i)
                    
                    
        for i in tmp3:
            if len(i) > 1:
                wechatid.append(i[0])
                displayname.append(i[1])
        
        matched_df = matched_df.append(pd.DataFrame({
                'chatroomname':chatroomname,
                'member':wechatid,
                'displayname':displayname
                }),ignore_index = True)
    #将得到的群昵称和微信昵称进行外链接
    matched_df = matched_df.merge(nicknamedf,on = ['chatroomname','member'],how = 'outer')
    return matched_df
        

nickname0830 = match_nickname(chatroom0830)
nickname0830 = match_groupdisplayname(chatroom0830,nickname0830)
#%%

#%%
'''
#get sender and msg
#在content中提取sender和发送的内容
'''
def get_sender_msg(msgdf):
    msgdf['sender'] = msgdf.content.str.split(':\n').str[0]
    msgdf['sender'][msgdf.type == 10000] = np.NaN
    msgdf['msg'] = msgdf.content.str.split(':\n',n=1).str[1]
#删除msgId列，因为没用,inplace的意思是是否在原始数据上进行操作
#删除content列，因为和sender，msg重复了
    msgdf.drop(columns = ['msgId','content'], inplace = True)
#%%


#%%
'''
 #msg filter modigy and combine
'''
message0830.createTime = pd.to_datetime(message0830.createTime,unit = 'ms')
message0830 = message0830[message0830.talker.isin(chatroom0830.chatroomname)]


get_sender_msg(message0830)
message0830 = message0830.merge(nickname0830,left_on = ['talker','sender'],right_on = ['chatroomname','member'],how = 'left')
#所以说total_msg中是将displayname和nickname整合了的
total_msg = message0830[['msgSvrId','type','status','isSend','createTime','talker','sender','displayname','nickname','msg']]
#%%


#%%
'''
#另一个手机上的备份，从10月到现在
#以后的聊天记录备份都可以在该部分处理
'''
message180119 = pd.read_csv(r'../records/180119/message.csv')
chatroom180119 = pd.read_csv(r'../records/180119/chatroom.csv')
appmsg180119 = pd.read_csv(r'../records/180119/appmessage.csv')
appinfor180119 = pd.read_csv(r'../records/180119/appinfor.csv')
#180213是180119的增量更新，可以直接覆盖
message180213 = pd.read_csv(r'../records/180213/message.csv')
chatroom180213 = pd.read_csv(r'../records/180213/chatroom.csv')
appmsg180213 = pd.read_csv(r'../records/180213/appmessage.csv')
appinfor180213 = pd.read_csv(r'../records/180213/appinfo.csv')
#180310不是上一次的增量更新，需要和之前的合并
message180310 = pd.read_csv(r'../records/180310/message.csv')
chatroom180310 = pd.read_csv(r'../records/180310/chatroom.csv')
appmsg180310 = pd.read_csv(r'../records/180310/appmessage.csv')
appinfor180310 = pd.read_csv(r'../records/180310/appinfor.csv')

#%%

#%%
'''
chatroom filter 
'''
chatroom180119 = chatroom180119[~chatroom180119.chatroomname.isin(nokidswantgroup)]
message180119.talker[message180119.content.str.contains('"移出群聊').fillna(False)]
remove_room = pd.concat([remove_room,message180119.talker[message180119.content.str.contains('"移出群聊').fillna(False)]],ignore_index = True).drop_duplicates().reset_index(drop = True)

chatroom180310 = chatroom180310[~chatroom180310.chatroomname.isin(nokidswantgroup)]
message180310.talker[message180310.content.str.contains('"移出群聊').fillna(False)]#2月份被6个群移除了群聊
#和之前的remove_room合并
remove_room = pd.concat([remove_room,message180310.talker[message180310.content.str.contains('"移出群聊').fillna(False)]],ignore_index = True).drop_duplicates().reset_index(drop = True)
#这里的remove_room是第一个的子集
#所以直接使用
chatroom180119 = chatroom180119[~chatroom180119.chatroomname.isin(remove_room)]
chatroom180310 = chatroom180310[~chatroom180310.chatroomname.isin(remove_room)]
#%%

#%%
'''
get the nickname and group display name
'''
nickname180119 = match_nickname(chatroom180119)
nickname180119 = match_groupdisplayname(chatroom180119,nickname180119)
nickname180213 = nickname180119#2月份的记录使用的1月份的chatroom

nickname180310 = match_nickname(chatroom180310)
nickname180310 = match_groupdisplayname(chatroom180310, nickname180310)
#%%


#%%
'''
message filter modify and combine
'''
message180213 = message180213[message180213.talker.isin(chatroom180119.chatroomname)]#其实此刻有两个群被移除，但是移除的时间并不长，所以不做处理
get_sender_msg(message180213)
#将display name merge到message上
message180213 = message180213.merge(nickname180213,left_on = ['talker','sender'],right_on = ['chatroomname','member'],how = 'left')
message180213.createTime = pd.to_datetime(message180213.createTime,unit='ms')
total_msg = pd.concat([total_msg,message180213],join='inner',ignore_index = True)

message180310 = message180310[message180310.talker.isin(chatroom180310.chatroomname)]
get_sender_msg(message180310)
#将display name merge到message上
message180310 = message180310.merge(nickname180310,left_on = ['talker','sender'],right_on = ['chatroomname','member'],how = 'left')
message180310.createTime = pd.to_datetime(message180310.createTime, unit = 'ms')
total_msg = pd.concat([total_msg,message180310],join='inner',ignore_index = True)
#%%


#按照msgSvrId进行去重
#这个操作一定要在从模拟器导入数据之前，因为这些数据没有SvrId
total_msg.drop_duplicates('msgSvrId',inplace = True)

#%%
'''
#解决聊天中的@问题
#这个方法不适合通过软件导出的记录
'''
#displaynames匹配wechat id
tmpdf = pd.DataFrame()
for room in displaynames.roomid.unique():
    members = chatroom180310[chatroom180310.chatroomname == room].memberlist.tolist()[0].split(';')
    members = pd.Series(members)
    '''
    join : {‘inner’, ‘outer’}, default ‘outer’. How to handle indexes on other axis(es). Outer for union and inner for intersection.
ignore_index : boolean, default False. If True, do not use the index values on the concatenation axis. The resulting axis will be labeled 0, ..., n - 1. This is useful if you are concatenating objects where the concatenation axis does not have meaningful indexing information.
!!! Note the index values on the other axes are still respected in the join.
So what i need to do is reindexing the sub displaynams dataframe to 0.1.2... and set join to inner, cause now it is out join and that's awarked
    '''
    tmp = pd.concat([displaynames[displaynames.roomid == room].reset_index(drop = True),members],axis = 1, ignore_index = True)
    tmpdf = pd.concat([tmpdf,tmp],ignore_index = True)
    #end of for
displaynames = tmpdf.dropna()
displaynames.columns = ['displayname','roomname','name','roomid','member']

#@人名通过unicode中的\u2005提取
re_unicode = re.compile(u'@(?P<atname>.*?)\u2005')

#msgat = total_msg.msg.str.decode('utf-8').str.extractall(re_unicode)#102397 py2
msgat = total_msg.msg.str.extractall(re_unicode)#102397 在py3中文本总是unicode不用担心编码问题，只需要在读取的时候指明编码方式即可
msgat.index.levels[0].name = 'msgindex'
total_msg.index.name = 'msgindex'
# join with index, but the index should be named first
msgat = msgat.join(total_msg[['msgSvrId','createTime','talker','sender']])#102397

#msgat.atname = msgat.atname.str.encode('utf-8') py3中不再需要
msgat = msgat.merge(displaynames,left_on = ['atname','talker'],right_on = ['displayname','roomid'],how = 'left')#102397
#只需要member和displayname的对照
msgat = msgat[['msgSvrId','atname','createTime','talker','member','sender']]
msgat.columns = ['msgSvrId','atname','createTime','talker','member_x','sender']
'''
在和displaynames表匹配完后，和nicknames表的匹配可以用如下的函数进行
分别匹配displaynam和nickname
'''
def concatNames(atdf,nicknamedf):
    #match the displayname
    atdf = atdf.merge(nicknamedf,left_on = ['atname','talker'],right_on = ['displayname','chatroomname'],how = 'left')
    atdf.member_x[atdf.member_x.isna()] = atdf.member[atdf.member_x.isna()]
    atdf = atdf[['msgSvrId','atname','createTime','talker','member_x','sender']]
    #match the nickname
    atdf = atdf.merge(nicknamedf,left_on = ['atname','talker'],right_on = ['nickname','chatroomname'],how = 'left')
    atdf.member_x[atdf.member_x.isna()] = atdf.member[atdf.member_x.isna()]
    atdf = atdf[['msgSvrId','atname','createTime','talker','member_x','sender']]
    return atdf
    
msgat = concatNames(msgat,nickname0830)
msgat = concatNames(msgat,nickname180119)
msgat = concatNames(msgat,nickname180310)

len(msgat[msgat.member_x.isna()])*1.0/len(msgat)#有53%的找不到@对应的人
#如果找不到displayname对应的wechatid，则删掉
msgat.dropna(inplace = True, subset = ['member_x'])
nickname0830.dropna(subset = ['displayname']).pipe(lambda x: x.loc[x.displayname.str.contains('赛')])
nickname0830.dropna(subset = ['nickname']).pipe(lambda x: x.loc[x.nickname.str.contains('赛')])
#msgat data persistence
msgat.to_pickle(r'../records/sample/msgat_dataframe')
'''
#ToDO
用聊天日志中的@名称去匹配displaynames，发现有很多找不到（6701个昵称找不到），试图通过昵称和房间号去nicknames的四个表中去寻找；
但是，通过模拟器取得的那些聊天不需要通过这种方法，因为本身记录的就是群昵称，先通过这部分进行算法的训练
但是，软件导出的聊天记录也不是全部都有displayname，还是残缺的
'''
def find_displaynames(df):
    #for row in df.itertuples():
        #if(row.createTime )
    return 0

#2017-12-07 04:00:48.001000

#%%

#从模拟器中导出的聊天记录
#from '2017-08-30 12:27:50' to '2017-10-21 19:17:49'
#导出的文件含有乱码，这个问题如何解决
#%%
msg_from_simul = pd.read_csv(r'../records/171021/total_msg(2).csv')
msg_from_simul.columns = ['talker','msg','createTime','sender']

msg_from_simul['displayname'] = msg_from_simul.sender.str.extract('^(.*?)\(', expand = False)
msg_from_simul.sender = msg_from_simul.sender.str.extract('.*\\((.*)\\).*', expand = False)
msg_from_simul.createTime = pd.to_datetime(msg_from_simul.createTime)
#这部分的中文displayname含有乱码，先不使用
#total_msg = pd.concat([total_msg,msg_from_simul],join='outer',ignore_index = True)
#%%

#数据整合工作完成！！！


#%%
#观察每个月的活跃度
#total_msg.groupby(total_msg.createTime.dt.month+total_msg.createTime.dt.year*100).size()
#%%

#%%
#将2018年2月份（含）的聊天记录输出到csv
#这里需要注意，最后需要的可能只是所有完整的log，也就是说要使用最新的chatroom进行筛选
total_msg[(total_msg.createTime < '2018-03-01')&(total_msg.talker.isin(chatroom180310.chatroomname))].to_pickle(r'../records/sample/sample_msg_pickle')
#%%


