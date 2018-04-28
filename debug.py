w_contextFree = 0.14#0.45

w_auther = 0.3#0.3#0.15
w_conver = 0.6#0.6#0.8
w_temp = 0.1#0.1#0.05
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
'''Compute the similarity between a given msg and existing threads
Single pass clustering is performed.(reference about single pass tech: 
    http://facweb.cs.depaul.edu/mobasher/classes/csc575/assignments/single-pass.html)
1.Treat the first msg as a single-message cluster T.
2.for each remaining msg m compute for All the existing threads.
    sim(m, T)= max sim(m, mi) mi belongs to All the existing threads.
    sim(m, mi) = cosin similarity between these two msgs.
'''
def dictNorm(dict1):
    """Compute the norm of a dictionary just like a vector
    Args:
        dict1(dict): the square of sum of each value in this dict.
    Returns:
        The return float.
    """
    return(sum([v*v for v in dict1.values()]))
def dotProduct(dict1,dict2):
    """Compute the dot product of two dicts, like vectors
    Args:
        dict1, dict2(dicts): two tfidf dicts.
    Returns:
        The return float.
    """
    return sum([dict1[k] * dict2[k] for k in dict1.keys() & dict2.keys()])

def similarity(msgdf, targetmsgid, msgdate, threadDict, threshold):
    """Pairwise similarity function.

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
    """
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
            print('new thread!',max_similarity)
            if threaddate == msgdate:
                threadDict[max(threadDict.keys())%10000 + 1 + msgdate*10000] = [targetmsgid]
            if threaddate < msgdate:
                #if this a new date and new thread, then reset the thread id to 1.
                threadDict[1 + msgdate*10000] = [targetmsgid]
    return threadDict

threadDict = {}
begin_date = min(ann2['createTime'])
threshold = 0.6#0.7
# Use count to control how many msgs to distengle thread.
count = 0
for row in ann2.itertuples():
    msgid = row[2]
    msgdate = row[4]
    msgdate = (msgdate - begin_date).days + 1
    threadDict = similarity(ann2, msgid, msgdate, threadDict, threshold)
    count += 1
    #debug
    #if count >2: break
#for k,v in threadDict.items():
#    print(k, len(v))

# compare the performance between for loop map() and list comprehension
# map is not suitable for this iteration.
# and list comprehension is not suitable too.


#%%
'''Choose the F value to be the object and try to train the optimal model.
1. Build the F value.
2. Adjust the parameters to maximize F value.
'''

def fvalue(_realThreadDict, _threadDict):
    """The whole F measure of the detection result in a stream
is defined as a weighted sum over all threads as follow.
    Args:
        _realThreadDict(dict): The dict of ground true threads of msgs. Key is thread id and value is msgid list.
        _threadDict(dict): The dict of detected thread.
    """

    def recall(ti, tj):
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
        realMsg = _realThreadDict[ti]
        detectMsg = _threadDict[tj]
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
        realMsg = _realThreadDict[ti]
        detectMsg = _threadDict[tj]
        nij = len([real for real in realMsg if real in detectMsg])
        nj = len(detectMsg)
        # check the result with jupyter console
        # good result
        return nij / nj

    def pairf(ti,tj):
        """ F(i,j) = 2*Precision*Recall /(Precision + Recall)
        is the F measure of detected thread j and the real thread i.
        """
        prevalue = precision(ti, tj)
        revalue = recall( ti,  tj)
        # print(prevalue, revalue)
        if (prevalue == 0) | (revalue == 0):
            return 0
        else:
            return 2 * prevalue * revalue / (prevalue + revalue)

    max_pairf, wholef = 0, 0
    len_msg = len(ann2)
    for realThread in _realThreadDict:
        max_pairf = 0
        for detectThread in _threadDict:
            value_pairf = pairf(realThread, detectThread)
            if value_pairf > max_pairf: max_pairf = value_pairf
        wholef = wholef + len(_realThreadDict[realThread])*max_pairf

    return wholef/len_msg

realThreadDict = {}
for thread in ann2['thread'].unique():
    msgs = ann2[ann2['thread'] == thread]['msgSvrId'].tolist()
    realThreadDict[thread] = msgs

fv = fvalue(realThreadDict, threadDict)
#%% TEST and PLOT
def plotThreadNum(realThreadDict, _threadDict):
    realThreadTuple = [(k, realThreadDict[k]) for k in sorted(realThreadDict,key= lambda x: len(realThreadDict[x]))]
    threadTuple = [(k, threadDict[k]) for k in sorted(_threadDict,key= lambda x: len(_threadDict[x]))]

    p1 = plt.bar(list(range(len(realThreadTuple))),[len(v) for k,v in realThreadTuple])
    p2 = plt.bar(list(range(len(threadTuple))),[-len(v) for k,v in threadTuple])
    # w_contextFree here is a global variable!
    ptitle = 'fval_'+str(int(fvalue(realThreadDict, _threadDict)*10000))+'w_contextfree'+str(int(w_contextFree*100))+\
             'w_conver'+str(int(w_conver*100))+'w_auther'+str(int(w_auther*100))+'w_tmp'+str(int(w_temp*100))+'T'+str(int(threshold*100))
    plt.title(ptitle)
    plt.legend((p1[0], p2[0]), ('realThread', 'detectedThread'))

    plt.ylabel('# of MSGs')
    plt.xlabel('Threads')
    plt.show()
    plt.savefig(r'../images/'+ptitle)
plotThreadNum(realThreadDict, threadDict)