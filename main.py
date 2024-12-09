import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
import hashlib
from NetflixSimiarlity import NetflixSimiarlity
import time
from scipy.sparse import csr_matrix, csc_matrix
import matplotlib.pylab as plt


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
    print("step 1: data has been read")
    start_time = time.time()
    data = np.load('user_movie_rating.npy')
    user_movie_csr = create_user_movie_matrix(data)
    # Now you can save it or perform operations as needed
    # user_movie = load_npz("./user_movie.npz")
    movie_user_csr = user_movie_csr.transport()
    del data, user_movie_csr # delet the unnecessary variables to save memory
    print("step 2: user_movie_matrix has been obtained & NetflixSimiarlity_user start")
    NetflixSimiarlity_user = NetflixSimiarlity(movie_user_csr)
    print("step 3: NetflixSimiarlity_user has been initalized & signature matrix obatining")
    NetflixSimiarlity_user.create_signature_matrix_sparse_parallel(num_permutations = 10)
    print("step 4: signature matrix has been obtained & candidate pairs obtaining")
    NetflixSimiarlity_user.bands_hashing(bandNum=1, rowNum=2)
    print("step 5: candidate pairs has been obtained & Jaccard similarity computing")
    #print(NetflixSimiarlity_user.candidate_pairs)
    filtered_Jaccard = NetflixSimiarlity_user.Jaccard_simiarlity(threshold = 0.5)
    print("step 6: Jaccard similarities has been obtained")
    end_time = time.time()
    print(len(filtered_Jaccard))
    ordered_filter_Jaccard = sorted([item[-1] for item in filtered_Jaccard])

    plt.scatter(range(len(filtered_Jaccard)), ordered_filter_Jaccard, colour = "blue")
    plt.xlabel("Most similar pairs")
    plt.ylabel("Last values")
    plt.show()


    with open ("result.txt", "w") as file:
        for item in filtered_Jaccard:
            file.write(f"{item}\n")

    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.4f} seconds")

    print("Everything is done")
