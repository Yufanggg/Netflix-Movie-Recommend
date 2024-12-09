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

    def process_column_sparse(self, permutation):
        """
        this function intends to use the sparse matrix
        """

        # Conduct the row-based permutation directly on the CSR matrix
        permutated_user_movie_sparse_csr = self.user_movie_sparse[permutation, :]
        row_indices, col_indices = permutated_user_movie_sparse_csr.nonzero()
        unique_col_indices = np.unique(col_indices)    
        sorted_smallest_row_indices = [np.min(row_indices[col_indices == col]) for col in unique_col_indices]
        
        return sorted_smallest_row_indices


    
    def create_signature_matrix_sparse_parallel(self, num_permutations = 100):
        """
        this function intends to obtain the signature matrix from the user_movie_matrix
        """
        num_users = self.user_movie_sparse.shape[0]
        permutations = np.array([np.random.permutation(num_users) for _ in range(num_permutations)])

        signature_matrix = Parallel(n_jobs=-1, backend='threading')(
            delayed(self.process_column_sparse)(permutation) for permutation in permutations
            )
        
        self.signature_matrix_csr = csr_matrix(signature_matrix)

    def process_band(self, band_signature_matrix_csr):
        """
        Processes one band and returns a local hash table for that band.
        """

        local_hash_table = defaultdict(list)
        for col_index in range(band_signature_matrix_csr.shape[1]):
            col_data = band_signature_matrix_csr[:, col_index]
            band_signature_col = (tuple(col_data.data), tuple(col_data.indices))
            hash_value = hash(band_signature_col)
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
            end_row = (band + 1) * rowNum if band < (bandNum - 1) else self.signature_matrix_csr.shape[0]

            # Get the sub-matrix for the current band
            band_signature_matrix_csr = self.signature_matrix_csr[start_row: end_row,:]
            local_hash_table = self.process_band(band_signature_matrix_csr)

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

# creator.bands_hashing(bandNum=3, rowNum=1)
# print(creator.candidate_pairs)
# print(creator.Jaccard_simiarlity(threshold=0))





