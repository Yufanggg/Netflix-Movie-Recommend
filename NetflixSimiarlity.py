# all necessary packages
import numpy as np
import hashlib
from collections import defaultdict
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

from multiprocessing import Pool


class NetflixSimiarlity:
    def __init__(self, user_movie_matrix):
        self.user_movie_matrix = user_movie_matrix
    

    def process_column_sparse(self, col, permutation):
        """
        this function intends to use the sparse matrix
        """  
        # print(col, "==="*10, permutation)
        # Get the column as a dense array
        column_data = self.user_movie_matrix[:, col]
        # print("column_data:", column_data)
        # Convert to a sparse column in COO format
        column = coo_matrix(column_data).tocoo()
        
        # print("column:", column) 
        if column.nnz > 0:      
            row_indices = column.col # get the row indices of the non-zero entries in the column
            # print(row_indices)
            # print (np.min(permutation[row_indices]))
            return np.min(permutation[row_indices])
        return np.inf
    
    def create_signature_matrix_sparse_parallel(self, num_permutations = 1000):
        """
        this function intends to obtain the signature matrix from the user_movie_matrix
        """
        num_users, num_movies = self.user_movie_matrix.shape
        permutations = np.array([np.random.permutation(num_users) for _ in range(num_permutations)])
        # results = []
        # for col in range(num_movies):
        #     column_results = []
        #     for permutation in permutations:
        #         column_results.append(self.process_column_sparse(col, permutation))
        #     results.append(column_results)

        # signature_matrix = np.array(results).T
        results = np.array(Parallel(n_jobs=-1, backend='threading')(
            delayed(self.process_column_sparse)(col, permutation) 
            for col in range(num_movies) 
            for permutation in permutations
            )).T
        signature_matrix = results.reshape(num_permutations, num_movies)
        self.signature_matrix = signature_matrix

    def bands_hashing(self, bandNum):
        """
        This function intends to obtain the possible similarity columns via LHS out of the signature matrix
        bandNum and rowNum is the way to partiate the signature matrix. row number of the signature matrix = bandNum * rowNum
        """

        hash_tables = [defaultdict(list) for _ in range(bandNum)]  # One hash table per band
        rowNum = self.signature_matrix.shape[0]//bandNum
        if self.signature_matrix.shape[0] % bandNum != 0:         
            raise ValueError("The total number of rows in signature matrix must be divisible by bandNum.")

        for band in range(bandNum):
            start_row = band * rowNum
            end_row = (band + 1) * rowNum
            band_signature_matrix = self.signature_matrix[start_row: end_row,]

            for col_index in range(self.signature_matrix.shape[1]):
                band_signature = band_signature_matrix[:, col_index].tobytes()  # Create a tuple representing the signature for this item
                hash_value = hashlib.md5(band_signature).hexdigest()  # Use a hash function (e.g., Python's built-in hash function)
                # Add the column (item index) to the hash bucket for this band
                hash_tables[band][hash_value].append(col_index)
            
        # find the candidate pairs
        candidate_pairs = set()
        for band, hash_table in enumerate(hash_tables):
            for bucket in hash_table.values():
                if len(bucket) > 1:
                    # Generate all pairs of items in this bucket
                    for i in range(len(bucket)):
                        for j in range(i + 1, len(bucket)):
                            candidate_pairs.add((bucket[i], bucket[j]))
                            
        # Output candidate pairs
        self.candidate_pairs = candidate_pairs

    def Jaccard_simiarlity(self, threshold = 0.5):
        Jaccard_simiarlity = []
        for (col_1, col_2) in self.candidate_pairs:
            obj_1, obj_2 = self.user_movie_matrix[:, col_1], self.user_movie_matrix[:, col_2]
            # print(obj_1, obj_2)
            interction = np.sum(np.logical_and(obj_1, obj_2))
            union = np.sum(np.logical_or(obj_1, obj_2))
            Jaccard = interction/union
            Jaccard_simiarlity.append((col_1, col_2, Jaccard))

        filtered_Jaccard = [tup for tup in Jaccard_simiarlity if tup[2] > threshold]
        return(filtered_Jaccard)



# # testing code for the signature_matrix
# np.random.seed(123456)
# user_movie_matrix = np.array([
#     [1, 0, 1, 0],  # User 1 rated Movie 1 and Movie 3
#     [0, 1, 0, 1],  # User 2 rated Movie 2 and Movie 4
#     [1, 0, 0, 1],  # User 3 rated Movie 1 and Movie 4
#     [0, 1, 1, 0],  # User 4 rated Movie 2 and Movie 3
#     [1, 1, 0, 0],  # User 5 rated Movie 1 and Movie 2
# ])

# # Create an instance of SignatureMatrixCreator
# creator = NetflixSimiarlity(user_movie_matrix)

# # Create the signature matrix with 3 permutations
# creator.create_signature_matrix_sparse_parallel(num_permutations = 10)
# # creator.create_signature_matrix_sparse_parallel(num_permutations=10)
# print("&"*30)
# print(creator.signature_matrix.shape[0])
# print(creator.signature_matrix)

# creator.bands_hashing(bandNum=10)
# print(creator.candidate_pairs)
# print(creator.Jaccard_simiarlity(threshold=0))





