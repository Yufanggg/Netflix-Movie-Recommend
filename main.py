
import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
import hashlib
from NetflixSimiarlity import NetflixSimiarlity
import time


def create_user_movie_matrix(data):
    # get lenth of users and movie
    # get lenth of users and movie
    has_nan = np.isnan(data).any()
    if has_nan:
        print("The array contains NaN values.")
    else:
        print("The array does not contain any NaN values.")
    
    userlen=len(np.unique(data[:,0])) + 1
    movielen=len(np.unique(data[:,1])) + 1
    # print(userlen,movielen)
    user = data[:,0]
    movie = data[:,1]
    Rating = np.ones_like(data[:,2])# make the value become binary
    user_movie = coo_matrix((Rating, (user, movie)), shape = (userlen, movielen))
    #user_movie_matrix = user_movie.toarray()

    save_npz("user_movie.npz", user_movie)
    return user_movie

# convert  value in every row to a 128 array ,then get a signature matrix
# def minhash(data,seed):
#     similaruser=[]

#     for row in data:
#         m=Minhash(seed=seed)
#         # upodate hasevalue in one row
#         for index in row.indices:
#             m.update(str(index).encode('utf-8'))
#         similaruser.append(m)
#     return similaruser

#  a b
# a[1,2
# b 1,2]

if __name__ == '__main__':
    np.random.seed(123456)
    start_time = time.time()
    data = np.load('user_movie_rating.npy')
    user_movie = create_user_movie_matrix(data)
    user_movie_matrix = user_movie.toarray()
    # Now you can save it or perform operations as needed
    # user_movie = load_npz("./user_movie.npz")

    print("this is my user_movie_matrix")
    print(user_movie_matrix)
    movie_user_matrix = user_movie_matrix.T
    del data, user_movie, user_movie_matrix # delet the unnecessary variables to save memory
    NetflixSimiarlity_user = NetflixSimiarlity(movie_user_matrix)
    NetflixSimiarlity_user.create_signature_matrix_sparse_parallel(num_permutations = 1000)
    NetflixSimiarlity_user.bands_hashing(bandNum=1)
    print(NetflixSimiarlity_user.candidate_pairs)
    filtered_Jaccard = NetflixSimiarlity_user.Jaccard_simiarlity(threshold = 0.5)
    end_time = time.time()
    print(filtered_Jaccard)
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.4f} seconds")

    