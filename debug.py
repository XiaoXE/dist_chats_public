from sklearn.metrics.pairwise import cosine_similarity
v2 = ann2['extended'].iloc[1]
v1 = ann2['extended'].iloc[0]
import time
start_time = time.time()
newv1 = [v1[k] for k in v1.keys() & v2.keys()]
newv2 = [v2[k] for k in v1.keys() & v2.keys()]

cosine_similarity([newv1,newv2])
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
dotProduct(v1,v2)
print("--- %s seconds ---" % (time.time() - start_time))

cosine_similarity([newv1[10:],newv2[10:]])