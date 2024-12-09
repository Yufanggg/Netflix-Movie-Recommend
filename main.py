import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
import hashlib
from NetflixSimiarlity import NetflixSimiarlity
import time
from scipy.sparse import csr_matrix, csc_matrix


def create_user_movie_matrix(data):
    # get lenth of users and movie
    # get lenth of users and movie
    has_nan = np.isnan(data).any()
    if has_nan:
        print("The array contains NaN values.")
    else:
        print("The array does not contain any NaN values.")
    
    userlen= np.max(data[:,0]) + 1
    movielen= np.max(data[:,1]) + 1
    # print(userlen,movielen)
    user = data[:,0]
    movie = data[:,1]
    Rating = np.ones_like(data[:,2])# make the value become binary
    user_movie = csr_matrix((Rating, (user, movie)), shape = (userlen, movielen))
    #user_movie_matrix = user_movie.toarray()
    return user_movie


if __name__ == '__main__':
    np.random.seed(123456)
    start_time = time.time()
    data = np.load('user_movie_rating.npy')
    user_movie_csr = create_user_movie_matrix(data)

    print("user_movie_matrix has been obtained")
    movie_user_csr = user_movie_csr.transpose()
    NetflixSimiarlity_user = NetflixSimiarlity(movie_user_csr)
    NetflixSimiarlity_user.create_signature_matrix_sparse_parallel(num_permutations = 10)
    NetflixSimiarlity_user.bands_hashing(bandNum=10, rowNum=1)
    #print(NetflixSimiarlity_user.candidate_pairs)
    filtered_Jaccard = NetflixSimiarlity_user.Jaccard_simiarlity(threshold = 0.5)
    end_time = time.time()
    print(filtered_Jaccard)
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.4f} seconds")
