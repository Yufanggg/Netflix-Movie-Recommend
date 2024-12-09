import sys

import numpy as np
from scipy.sparse import load_npz

data=np.load('user_movie_rating.npy')
# print(data[100])

# caluacte similarity
def Jaccard_Similarity(user1, user2):
    non_zero_user1 = user1 > 0
    non_zero_user2 = user2 > 0


    intersection = np.sum(non_zero_user1 & non_zero_user2)
    union = np.sum(non_zero_user1 | non_zero_user2)


    jaccard_similarity = intersection / union
    return jaccard_similarity

def get_data(data,userId):
    user_ratings = data[userId].toarray().flatten()

    return user_ratings

# user_movie_matrix=load_npz("user_movie.npz")
# user_movie_matrix=user_movie_matrix.tocsr()
# print(user_movie_matrix.shape)
# np.set_printoptions(threshold=np.inf)
# print(get_data(user_movie_matrix, 103704))
# data = np.load('user_movie_rating.npy')
# print(data[:10])
# user1=np.array([0,5,4,0,3,0])
# user2=np.array([5,4,3,0,0,0])
# user1 = np.array([0, 5, 4, 0, 3, 0])
# user2 = np.array([5, 4, 3, 0, 0, 0])
# print(Jaccard_Similarity(user1,user2))


