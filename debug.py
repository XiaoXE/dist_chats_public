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
    return(sum([dict1[k]*dict2[k] for k in dict1.keys()&dict2.keys()]))

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
    if(len(threadDict) == 0):
        threadid = 1 + 10000*msgdate
        threadDict[threadid] = [targetmsgid]
    else:
        max_similarity_thread = 0
        max_similarity = 0
        #find the max similarity and the corresponding thread over all threads.
        for thread, msgids in threadDict.items():
            #Notice, the threadDict may be NULL!
            threaddate = thread // 10000
            if(threaddate < msgdate - 1): continue
            for msgid in msgids:
                targetmsg = msgdf[msgdf.msgSvrId == targetmsgid]['extended'].item()#get the exact dict rather than dict and object type
                comparedmsg = msgdf[msgdf.msgSvrId == msgid]['extended'].item()
                cosine = dotProduct(targetmsg,comparedmsg)/(dictNorm(targetmsg)*dictNorm(comparedmsg))
                if (cosine > max_similarity):
                    max_similarity = cosine
                    max_similarity_thread = thread
                     
        if(max_similarity > threshold):
            threadDict[max_similarity_thread].append(targetmsgid)
        else:
            #CAUTION maybe wrong
            #create a new thread
            if(threaddate == msgdate):
                threadDict[max(threadDict.keys())%10000 + 1 + msgdate*10000] = [targetmsgid]
            if(threaddate < msgdate):
                #if this a new date and new thread, then reset the thread id to 1.
                threadDict[1 + msgdate*10000] = [targetmsgid]
    return(threadDict)    
threadDict = {}       
begin_date = min(ann2['createTime'])
threshold = 0.7
count = 1
for row in ann2.itertuples():
    msgid = row[2]
    msgdate = row[4]
    msgdate = (msgdate - begin_date).days + 1
    threadDict = similarity(ann2, msgid, msgdate, threadDict, threshold)
    count += 1
    if(count >50): break