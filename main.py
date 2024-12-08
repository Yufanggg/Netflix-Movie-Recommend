
import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
import hashlib

from Lsh import MinHashLSH
from minhash import Minhash
from similarity import Jaccard_Similarity, get_data


def load_data(data):
    # get lenth of users and movie
    userlen=len(np.unique(data[:,0]))
    movielen=len(np.unique(data[:,1]))
    print(userlen,movielen)
    # 构建思路，读取每一行和每一列，[user,movid]=rate,使用稀疏矩阵
    rate, rows, cols = [], [], []
    for row in data:
        rows.append(row[0])
        cols.append(row[1])
        rate.append(1)
    user_movie = coo_matrix((rate, (rows, cols)), shape=(userlen+1, movielen+1))
    save_npz("user_movie.npz", user_movie)
    return user_movie

# convert  value in every row to a 128 array ,then get a signature matrix
def minhash(data,seed):
    similaruser=[]

    for row in data:
        m=Minhash(seed=seed)
        # upodate hasevalue in one row
        for index in row.indices:
            m.update(str(index).encode('utf-8'))
        similaruser.append(m)
    return similaruser

#  a b
# a[1,2
# b 1,2]

if __name__ == '__main__':
    seed=42
    # data = np.load('user_movie_rating.npy')
    # user_movie_matrix = load_data(data)
    user_movie_matrix=load_npz("user_movie.npz")
    user_movie_matrix=user_movie_matrix.tocsr()
    signature_matrix=minhash(user_movie_matrix,seed)
    print(signature_matrix[1].hashvalues)
    # minhashs={user+1:minhash for user,minhash in enumerate(signature_matrix)}
    minhashs = {user + 1: minhash for user, minhash in enumerate(signature_matrix)}
    # init lsh with b and r
    lsh=MinHashLSH(params=[8,16])
    # add hash value to different buckets
    for user,minhash in minhashs.items():
        lsh.insert(user,minhash)

    # candidates = {}
    candidates=[]
    for user,minhash in minhashs.items():
        similarlity_list=lsh.query(minhash)
        similarlity_list.remove(user)
        for row in similarlity_list:
            candidates.append(row)
    print(len(candidates))
        # candidates[user] = similarlity_list




    # print(candidates[1])
    # for i in range(1,10):
    #     print(candidates[i])
    # longest_key, longest_value = max(candidates.items(), key=lambda item: len(item[1]))
    # sim_list=[]
    # for key,val in candidates.items():
    #     if val !=None:
    #         for i in val:
    #             if(Jaccard_Similarity(get_data(user_movie_matrix,key),get_data(user_movie_matrix,i))) >0.5:
    #                 sim_list.append((key,i))
    # print(sim_list)
    # print(len(sim_list))
    # print(f"最长的键: {longest_key}")
    # print(f"最长的值: {longest_value}")
    # print(len(candidates))


    # print(signature_matrix[0])
    # print(len(signature_matrix))