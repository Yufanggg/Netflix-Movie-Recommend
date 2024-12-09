# all necessary packages
import numpy as np
import hashlib
import random
from collections import defaultdict
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from itertools import combinations
from tqdm import tqdm
from scipy.sparse import csr_matrix, csc_matrix

class NetflixSimiarlity:
    def __init__(self, user_movie_sparse):
        self.user_movie_sparse = user_movie_sparse
        self.num_user = self.user_movie_sparse.shape[0]
        self.num_movie = self.user_movie_sparse.shape[1]

    def process_column_sparse(self, permutation):
        """
        this function intends to use the sparse matrix
        """
        # conduct the row-base permutation
        permutated_user_movie_sparse= self.user_movie_sparse[permutation, :]
        # Convert the result back to CSR format if you need
        permutated_user_movie_sparse_coo = permutated_user_movie_sparse.tocoo()
        
        # get the indices of rows
        # print(permutated_user_movie_sparse_coo)
        row_indices, col_indices = permutated_user_movie_sparse_coo.row, permutated_user_movie_sparse_coo.col
        # print(row_indices, col_indices)
        
        smallest_row_indices = np.full(self.num_movie, float('inf'))
        # Iterate through the row and column indices
        for row_index, col_index in zip(row_indices, col_indices):    
            smallest_row_indices[col_index] = min(smallest_row_indices[col_index], row_index)
        # Filter out infinite values and get the smallest row index for each column
        sorted_smallest_row_indices = smallest_row_indices[smallest_row_indices != float('inf')]    
   
        return(sorted_smallest_row_indices)
    
    def create_signature_matrix_sparse_parallel(self, num_permutations = 100):
        """
        this function intends to obtain the signature matrix from the user_movie_matrix
        """
        
        permutations = np.array([np.random.permutation(self.num_user) for _ in range(num_permutations)])

        signature_matrix = Parallel(n_jobs=-1, backend='threading')(
            delayed(self.process_column_sparse)(permutation) for permutation in tqdm(permutations, desc="Processing")
            )

        self.signature_matrix = np.array(signature_matrix)


    def process_band(self, band_signature_matrix):
        """
        Processes one band and returns a local hash table for that band.
        """

        local_hash_table = defaultdict(list)
        for col_index in range(band_signature_matrix.shape[1]):
            band_signature = band_signature_matrix[:, col_index].tobytes()
            hash_value = hashlib.md5(band_signature).hexdigest()
            local_hash_table[hash_value].append(col_index)

        return local_hash_table


    
    def bands_hashing(self, bandNum, rowNum):
        """
        This function intends to obtain the possible similarity columns via LHS out of the signature matrix
        bandNum and rowNum is the way to partiate the signature matrix. row number of the signature matrix = bandNum * rowNum
        """
        
        # Process each band and hash columns
        candidate_pairs = set()
        for band in tqdm(range(bandNum), desc="Processing"):
            start_row = band * rowNum
            end_row = (band + 1) * rowNum if band < (bandNum - 1) else self.signature_matrix.shape[0]

            # Get the sub-matrix for the current band
            band_signature_matrix = self.signature_matrix[start_row: end_row,:]
            local_hash_table = self.process_band(band_signature_matrix)

            #  Extract candidate pairs from the hash table
            for bucket in local_hash_table.values():
                if len(bucket) > 1:
                    # Generate all pairs of items in this bucket
                    for pair in combinations(bucket, 2):
                        # print(pair)
                        candidate_pairs.add(pair)
 
        self.candidate_pairs = candidate_pairs


    def Jaccard_simiarlity(self, threshold = 0.5):
        Jaccard_simiarlity = []
        for (col_1, col_2) in self.candidate_pairs:
            obj_1, obj_2 = self.user_movie_sparse[:, col_1], self.user_movie_sparse[:, col_2]
            row_indices_obj_1, row_indices_obj_2 = obj_1.tocoo().row, obj_2.tocoo().row
            # print(row_indices_obj_2, row_indices_obj_1)

            set1, set2 = set(row_indices_obj_1), set(row_indices_obj_2)
            interction = set1 & set2
            union = set1 | set2
            Jaccard = len(interction)/len(union)
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
# creator = NetflixSimiarlity(csr_matrix(user_movie_matrix))

# # Create the signature matrix with 3 permutations
# creator.create_signature_matrix_sparse_parallel(num_permutations=3)
# # print("&"*30)
# # print(creator.signature_matrix)

# creator.bands_hashing(bandNum=3, rowNum = 1)
# print(creator.candidate_pairs)
# print(creator.Jaccard_simiarlity(threshold=0))





