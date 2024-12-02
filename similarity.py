import numpy as np

data=np.load('user_movie_rating.npy')
print(data[100])

# caluacte similarity
def Jaccard_Similarity(user1, user2):
    ins = 0
    un_len=0
    for i,j in zip(user1,user2):
        if i !=0 and j!=0:
            ins+=1
        if i!=0 or j!=0:
            un_len+=1
    similarity=ins/un_len
    return similarity

user1=(0,5,4,0,3,0)
user2=(5,4,3,0,0,0)
print(Jaccard_Similarity(user1,user2))


