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
    def __init__(self, user_movie_sparse, seed = 42):
        self.user_movie_sparse = user_movie_sparse
        self.num_user = self.user_movie_sparse.shape[0]
        self.num_movie = self.user_movie_sparse.shape[1]
        self.random_state = np.random.RandomState(seed)  

    def generate_permutations(self, num_permutations: int):
        """
        Generator for user index permutations.
        
        Args:
            num_permutations (int): Number of permutations to generate.
            
        Yields:
            np.ndarray: A single user index permutation.
        """
        for _ in range(num_permutations):
            yield self.random_state.permutation(self.num_user)

    

    def process_column_sparse(self, permutation):
        """
        this function intends to use the sparse matrix
        """

        # Conduct the row-based permutation directly on the CSR matrix
        permutated_user_movie_sparse_csr = self.user_movie_sparse[permutation, :]
        row_indices, col_indices = permutated_user_movie_sparse_csr.nonzero()

        # Efficient computation of smallest row indices for each column
        unique_col_indices, first_occurrence_indices = np.unique(col_indices, return_index=True)
        smallest_row_indices = row_indices[first_occurrence_indices]
        
        return smallest_row_indices


    
    def create_signature_matrix_sparse_parallel(self, num_permutations = 100):
        """
        this function intends to obtain the signature matrix from the user_movie_matrix
        """

        permutation_generator = self.generate_permutations(num_permutations)
        # Process permutations in parallel to create the signature matrix
        signature_matrix = Parallel(n_jobs=-1)(
            delayed(self.process_column_sparse)(permutation) 
            for permutation in tqdm(permutation_generator, total=num_permutations, desc="Computing the Signature Matrix")
        )
        
        self.signature_matrix_csr = csr_matrix(signature_matrix)

    def process_band(self, band_signature_matrix_csr):
        """
        Processes one band and returns a local hash table for that band.
        """

        local_hash_table = defaultdict(list)
        for col_index in range(band_signature_matrix_csr.shape[1]):
            col_data = band_signature_matrix_csr[:, col_index].data
            hash_value = hash(col_data.tobytes())#, tuple(col_data.indices))
            local_hash_table[hash_value].append(col_index)

        return local_hash_table
        

    

    def bands_hashing(self, bandNum, rowNum):
        """
        This function intends to obtain the possible similarity columns via LHS out of the signature matrix
        bandNum and rowNum is the way to partiate the signature matrix. row number of the signature matrix = bandNum * rowNum
        """
        
        # Process each band and hash columns
        candidate_pairs = set()
        for band in tqdm(range(bandNum), desc="Computing the band hashing"):
            start_row = band * rowNum
            end_row = (band + 1) * rowNum 

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

    def Jaccard_simiarlity(self, pair): 
        col_1, col_2 = pair
        obj_1, obj_2 = self.user_movie_sparse[:, col_1], self.user_movie_sparse[:, col_2]
        
        # Nonzero row indices for each column (direct sparse access)
        indices_1 = set(obj_1.indices)  # Faster than converting to COO
        indices_2 = set(obj_2.indices)
        
        # Compute Jaccard similarity
        intersection = len(indices_1 & indices_2)
        union = len(indices_1 | indices_2)
        similarity = intersection / union if union > 0 else 0
        
        return (col_1, col_2, similarity)


    def Jaccard_simiarlity_parallel(self, threshold = 0.5):
        Jaccards = Parallel(n_jobs=-1)(delayed(self.Jaccard_simiarlity)(pair) 
                                       for pair in tqdm(self.candidate_pairs, desc="Computing Jaccard Similarity"))
        
        filtered_Jaccard = [res for res in Jaccards if res[2] > threshold]
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
# print(creator.Jaccard_simiarlity_parallel(threshold=0))





