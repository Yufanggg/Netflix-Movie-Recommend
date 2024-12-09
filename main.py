
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

    # 构建思路，读取每一行和每一列，[user,movid]=rate,使用稀疏矩阵
    rate, rows, cols = [], [], []
    for row in data:
        rows.append(row[0])
        cols.append(row[1])
        rate.append(row[2])
    print(rows[-10:])
    user_movie = coo_matrix((rate, (rows, cols)))
    print(user_movie.shape)
    save_npz("user_movie.npz", user_movie)
    return user_movie

# convert  value in every row to a 128 array ,then get a signature matrix
def minhash(data,seed):
    similaruser=[]

    for row in data:
        m=Minhash(seed=seed,d=128)
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
    data = np.load('user_movie_rating.npy')
    user_movie_matrix = load_data(data)
    # print("1111")
    # user_movie_matrix=load_npz("user_movie.npz")
    user_movie_matrix=user_movie_matrix.tocsr()
    signature_matrix=minhash(user_movie_matrix,seed)
    # print(signature_matrix[1].hashvalues)
    # minhashs={user+1:minhash for user,minhash in enumerate(signature_matrix)}
    signature_matrix=signature_matrix[1:]
    print(len(signature_matrix))
    minhashs = {user : minhash for user, minhash in enumerate(signature_matrix,start=1)}
    print(len(minhashs))
    # init lsh with b and r
    lsh=MinHashLSH(d=128,params=[14,9])
    # add hash value to different buckets
    for user,minhash in minhashs.items():
        lsh.insert(user,minhash)

    # candidates = {}
    candidates=[]
    testdata={}
    for user,minhash in minhashs.items():
        similarlity_list=lsh.query(minhash)
        similarlity_list.remove(user)
        testdata[user] = similarlity_list
        for row in similarlity_list:
            candidates.append(row)
    print(len(candidates))





    # print(candidates[1])
    # for i in range(1,10):
    #     print(candidates[i])
    # longest_key, longest_value = max(candidates.items(), key=lambda item: len(item[1]))103704
    sim_list=[]
    for key,val in testdata.items():
        if val !=None:
            for i in val:
                z=Jaccard_Similarity(get_data(user_movie_matrix, key), get_data(user_movie_matrix, i))
                sim_list.append((key,i,z))
    with open("result.txt", "w") as f:
        for item in sim_list:
            f.write(str(item) + "\n")
    print(sim_list)
    print(len(sim_list))
    # print(f"最长的键: {longest_key}")
    # print(f"最长的值: {longest_value}")
    # print(len(candidates))


    # print(signature_matrix[0])
    # print(len(signature_matrix))