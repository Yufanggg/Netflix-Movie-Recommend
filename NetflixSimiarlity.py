# all necessary packages
import numpy as np
import hashlib
import random
from collections import defaultdict

class NetflixSimiarlity:
    def __init__(self, user_movie_matrix):
        self.user_movie_matrix = user_movie_matrix

    def create_signature_matrix(self, permutationNum = 1000):
        """
        this function intends to obtain the signature matrix from the user_movie_matrix
        """
        # apply permutation over user_movie_matrix
        num_users, num_movies = self.user_movie_matrix.shape    
        # Initialize the signature matrix with 'inf' values (we'll fill it with the smallest indices)
        signature_matrix = np.full((permutationNum, num_movies), np.inf, dtype=int)

        for i in range(permutationNum): 
            # Precompute the random permutations of user indices
            permuted_user_indices = np.random.permutation(num_users)
            # Permute the user indices and extract the corresponding rows
            permuted_matrix = self.user_movie_matrix[permuted_user_indices]

            # For each movie, find the smallest index of the user who rated it (value 1)
            for col in range(num_movies):
                # Find the indices of users who rated the movie (value 1)
                rated_users = np.where(permuted_matrix[:, col] == 1)[0]
                if rated_users.size > 0:
                    # Take the first user (smallest index) who rated the movie
                    signature_matrix[i, col] = rated_users[0]
        
        self.signature_matrix = signature_matrix

        # return signature_matrix
    
    def bands_hashing(self, bandNum, rowNum):
        """
        This function intends to obtain the possible similarity columns via LHS out of the signature matrix
        bandNum and rowNum is the way to partiate the signature matrix. row number of the signature matrix = bandNum * rowNum
        """

        hash_tables = [defaultdict(list) for _ in range(bandNum)]  # One hash table per band
        assert bandNum*rowNum == self.signature_matrix.shape[0]

        for band in range(bandNum):
            start_row = band * rowNum
            end_row = (band + 1) * rowNum
            band_signature_matrix = self.signature_matrix[start_row: end_row,]

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
                    for i in range(len(bucket)):
                        for j in range(i + 1, len(bucket)):
                            candidate_pairs.add((bucket[i], bucket[j]))
                            
        # Output candidate pairs
        self.candidate_pairs = candidate_pairs

    def Jaccard_simiarlity(self, threshold = 0.5):
        Jaccard_simiarlity = []
        for (col_1, col_2) in self.candidate_pairs:
            obj_1, obj_2 = self.user_movie_matrix[:, col_1], self.user_movie_matrix[:, col_2]
            interction = np.sum(np.logical_and(obj_1, obj_2))
            union = np.sum(np.logical_or(obj_1, obj_2))
            Jaccard = interction/union
            Jaccard_simiarlity.append((col_1, col_2, Jaccard))

        filtered_Jaccard = [tup for tup in Jaccard_simiarlity if tup[2] > threshold]
        return(filtered_Jaccard)



# testing code for the signature_matrix
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
# creator.create_signature_matrix(permutationNum=3)
# print("&"*30)
# print(creator.signature_matrix.shape[0])

# creator.bands_hashing(bandNum=3, rowNum=1)
# print(creator.candidate_pairs)
# print(creator.Jaccard_simiarlity(threshold=0))





