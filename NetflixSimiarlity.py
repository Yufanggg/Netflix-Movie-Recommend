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
        # Convert to CSC format (more efficient for column slicing)
        user_movie_sparse_csc = self.user_movie_sparse.tocsc()

        # conduct the row-base permutation
        permutated_user_movie_sparse_csc= user_movie_sparse_csc[permutation, :]
        # Convert the result back to CSR format if you need
        permutated_user_movie_sparse_coo = permutated_user_movie_sparse_csc.tocoo()
        
        # get the indices of rows
        # print(permutated_user_movie_sparse_coo)
        row_indices, col_indices = permutated_user_movie_sparse_coo.row,  permutated_user_movie_sparse_coo.col
        # print(row_indices, col_indices)
        
        # List comprehension to get the smallest row index for each unique column index
        sorted_smallest_row_indices = [min(row_index for row_index, col_index in zip(row_indices, col_indices) 
                                           if col_index == unique_col_index)
                                       for unique_col_index in sorted(set(col_indices))]
        # print(sorted_smallest_row_indices)
        return(sorted_smallest_row_indices)
    
    def create_signature_matrix_sparse_parallel(self, num_permutations = 100):
        """
        this function intends to obtain the signature matrix from the user_movie_matrix
        """
        num_users = self.user_movie_sparse.shape[0]
        permutations = np.array([np.random.permutation(num_users) for _ in range(num_permutations)])
        # signature_matrix = []
        # for permutation in permutations:
        #     signature_matrix.append(self.process_column_sparse(permutation))
        #     print(signature_matrix)

        signature_matrix = Parallel(n_jobs=-1, backend='threading')(
            delayed(self.process_column_sparse)(permutation) for permutation in permutations
            )

        self.signature_matrix = np.array(signature_matrix)

    def process_band(self, band, rowNum):
        """
        Processes one band and returns a local hash table for that band.
        """
        start_row = band * rowNum
        end_row = (band + 1) * rowNum
        band_signature_matrix = self.signature_matrix[start_row: end_row,:]

        local_hash_table = defaultdict(list)
        for col_index in range(self.signature_matrix.shape[1]):
            band_signature = band_signature_matrix[:, col_index].tobytes()
            hash_value = hashlib.md5(band_signature).hexdigest()
            local_hash_table[hash_value].append(col_index)

            return local_hash_table


    
    def bands_hashing(self, bandNum):
        """
        This function intends to obtain the possible similarity columns via LHS out of the signature matrix
        bandNum and rowNum is the way to partiate the signature matrix. row number of the signature matrix = bandNum * rowNum
        """

        rowNum = self.signature_matrix.shape[0]//bandNum
        if self.signature_matrix.shape[0] % bandNum != 0:         
            raise ValueError("The total number of rows in signature matrix must be divisible by bandNum.")
        
        hash_tables = [defaultdict(list) for _ in range(bandNum)]  # One hash table per band
    

        for band in range(bandNum):
            start_row = band * rowNum
            end_row = (band + 1) * rowNum
            band_signature_matrix = self.signature_matrix[start_row: end_row,:]

            for col_index in range(self.signature_matrix.shape[1]):
                band_signature = tuple(band_signature_matrix[:, col_index])  # Create a tuple representing the signature for this item
                hash_value = hash(band_signature)  # Use a hash function (e.g., Python's built-in hash function)
                # Add the column (item index) to the hash bucket for this band
                hash_tables[band][hash_value].append(col_index)
            
        # find the candidate pairs
        candidate_pairs = set()
        for band, hash_table in enumerate(hash_tables):
            for bucket in hash_table.values():
                if len(bucket) > 1:
                    # Generate all pairs of items in this bucket
                    candidate_pairs.update(combinations(bucket, 2))

                    # for i in range(len(bucket)):
                    #     for j in range(i + 1, len(bucket)):
                    #         candidate_pairs.add((bucket[i], bucket[j]))
                            
        # Output candidate pairs
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

# creator.bands_hashing(bandNum=3)
# print(creator.candidate_pairs)
# print(creator.Jaccard_simiarlity(threshold=0))




