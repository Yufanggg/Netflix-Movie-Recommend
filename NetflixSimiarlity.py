# all necessary packages
import numpy as np
import hashlib
import random

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

        return signature_matrix
    
    def bands_hashing(self, bandNum, rowNum):
        """
        This function intends to obtain the possible similarity columns via LHS out of the signature matrix
        bandNum and rowNum is the way to partiate the signature matrix. row number of the signature matrix = bandNum * rowNum
        """

        self.signature_matrix = self.create_signature_matrix()
        for band in range(bandNum):
            start_row = band * rowNum
            end_row = (band + 1) * rowNum
            band_signature_matrix = self.signature_matrix[start_row: end_row,]
            





# testing code for the signature_matrix
user_movie_matrix = np.array([
    [1, 0, 1, 0],  # User 1 rated Movie 1 and Movie 3
    [0, 1, 0, 1],  # User 2 rated Movie 2 and Movie 4
    [1, 0, 0, 1],  # User 3 rated Movie 1 and Movie 4
    [0, 1, 1, 0],  # User 4 rated Movie 2 and Movie 3
    [1, 1, 0, 0],  # User 5 rated Movie 1 and Movie 2
])

# Create an instance of SignatureMatrixCreator
creator = NetflixSimiarlity(user_movie_matrix)

# Create the signature matrix with 3 permutations
signature_matrix = creator.create_signature_matrix(permutationNum=3)

# Output the result
print("Signature Matrix:")
print(signature_matrix)





