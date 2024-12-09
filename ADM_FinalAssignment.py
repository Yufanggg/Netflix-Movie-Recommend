import numpy as np
import scipy.sparse as sp
import sys
import random
import hashlib
from itertools import combinations
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt



class MultiprocessingTaskExecutor:
    def __init__(self, func, num_workers):

        self.func = func
        self.num_workers = num_workers if num_workers else cpu_count()

    def execute_map(self, tasks):

        with Pool(processes=self.num_workers) as pool:
            results = pool.map(self.func, tasks)
        return results

    def execute_starmap(self, tasks):

        with Pool(processes=self.num_workers) as pool:
            results = pool.starmap(self.func, tasks)
        return results



# Loading & Preprocessing Data:
'''Preprocessing involves creating a sparse matrix with binary values, where actual ratings are converted to 1. 
This transformation focuses solely on whether a user has rated a movie, disregarding the rating itself.'''
def load_data(file_path):
    
    """Load and preprocess data into a sparse matrix."""
    data = np.load(file_path)

    user_ids = data[:, 0]
    movie_ids = data[:, 1]
    ratings = data[:, 2]

    num_users = user_ids.max()
    num_movies = movie_ids.max()
    
    # Create sparse matrix: binary values (1 if rated, 0 otherwise)
    user_movie_matrix = sp.csr_matrix(
        (np.ones_like(ratings), (user_ids, movie_ids)), shape=(num_users + 1, num_movies + 1)
    )

    return user_movie_matrix



# Generating random Permutations:
'''Generating 𝑛 random permutations of movie indices ensures that each row of the user-movie matrix is permuted 
in a unique and randomized order during the MinHash computation.'''
def generate_permutations(num_permutations, num_movies, seed):
    
    """Generate random permutations of movie indices."""
    random.seed(seed)
    movie_indices = np.arange(num_movies)
    permutations = [np.random.permutation(movie_indices) for _ in range(num_permutations)]
    
    return permutations



# Minhasing with parallel processing
def generate_user_signature(user_data, permutations):
    
    """Generate the Minhash signature for a single user."""
    user_signature = []
    for perm in permutations:
        permuted_user_rows = perm[user_data.indices]

        # Check if permuted_user_rows is empty
        if len(permuted_user_rows) == 0:
            user_signature.append(np.inf)  # Or any other default value for empty rows
        else:
            user_signature.append(np.min(permuted_user_rows))
        
    return user_signature


def minhash_signature_with_permutations_parallel(user_movie_matrix, permutations, num_workers):
    
    """Generate Minhash signatures in parallel."""
    num_users = user_movie_matrix.shape[0]

    # Prepare user data
    user_data_list = [user_movie_matrix.getrow(user) for user in range(num_users)]

    # Execute in parallel
    executor = MultiprocessingTaskExecutor(func=generate_user_signature, num_workers=num_workers)
    signatures = executor.execute_starmap([(user_data, permutations) for user_data in user_data_list])

    return np.array(signatures).T  # Transpose to get the expected shape



# LSH Banding Technique
'''We divide the signatures into n bands and split them into chunks and hash before putting them in buckets.
The idea is similar pairs will end up in the same bucket thanks to hashing.
The signatures are divided into 𝑛 bands, with each band containing a chunk of the signature. 
These chunks are then hashed and placed into buckets. 
The key idea is that similar pairs are likely to hash into the same bucket, 
facilitating the efficient identification of potential matches.'''
def hash_band(band_signature):
    """Hash a band signature using SHA-256 and return a bucket value."""
    return int(hashlib.sha256(str(band_signature).encode()).hexdigest(), 16)


def lsh_with_hashing(signature_matrix, num_bands, threshold):
    """Apply LSH with hashing and secondary similarity checks."""
    n_permutations, n_users = signature_matrix.shape
    rows_per_band = n_permutations // num_bands
    candidate_pairs = set()
    confirmed_pairs = set()
    
    for band in range(num_bands):
        start = band * rows_per_band
        end = (band + 1) * rows_per_band
        band_hash = {}
        
        for user in range(n_users):
            band_signature = tuple(signature_matrix[start:end, user])
            bucket = hash_band(band_signature)  # Hash the band signature
            
            if bucket in band_hash:
                for other_user in band_hash[bucket]:
                    candidate_pairs.add((min(user, other_user), max(user, other_user)))
                band_hash[bucket].append(user)
            else:
                band_hash[bucket] = [user]
    
    return candidate_pairs


# Secondary Similarity Check in parallel:
'''Secondary check is performed to make sure whether the candidate pairs are really similar pairs''' 
def check_similarity_parallel(task):
    
    """Wrapper for secondary similarity check."""
    user1, user2, signature_matrix, threshold = task
    n_permutations = signature_matrix.shape[0]
    common_signatures = np.sum(signature_matrix[:, user1] == signature_matrix[:, user2])
    similarity = common_signatures / n_permutations

    return (user1, user2) if similarity >= threshold else None


def filter_similar_users(candidate_pairs, signature_matrix, threshold, num_workers):
    
    """Filter candidate pairs in parallel."""
    tasks = [(user1, user2, signature_matrix, threshold) for user1, user2 in candidate_pairs]
    
    executor = MultiprocessingTaskExecutor(func=check_similarity_parallel, num_workers=num_workers)
    results = executor.execute_map(tasks)
    
    # Remove None results
    similar_users = [pair for pair in results if pair is not None]
    return similar_users



# Jaccard Similarity
def jaccard_similarity(u1, u2, user_movie_matrix):
    
    """Compute Jaccard similarity between two users."""
    movies_u1 = set(user_movie_matrix[u1].indices)
    movies_u2 = set(user_movie_matrix[u2].indices)

    intersection = len(movies_u1 & movies_u2)
    union = len(movies_u1 | movies_u2)
    
    return intersection / union if union != 0 else 0



# Plot Function
def plot_similarity_scores(similarity_scores):

    # Extract the similarity scores
    scores = [score for _, _, score in similarity_scores]

    # Plot the similarity scores
    plt.figure(figsize=(20, 6))
    plt.plot(scores, marker='o', linestyle='-', color='orange')
    plt.title('LSH Similarity Scores of Similar User Pairs')
    plt.xlabel('Pair Index')
    plt.ylabel('Similarity Score')
    plt.grid(True)
    plt.show()



#Main
if __name__ == "__main__":
    
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        print("Command-line argument not provided or invalid in Colab.")
        seed = int(input("Enter the random seed: "))
    else:
        seed = int(sys.argv[1])
    
    output_file = "result.txt"
    data_file = "/content/drive/MyDrive/Colab Notebooks/ADVANCES IN DATA MINING/FINAL ASSIGNMENT/Dataset/user_movie_rating.npy"

    # Load data
    user_movie_matrix = load_data(data_file)
    num_users, num_movies = user_movie_matrix.shape
    print(f"Total users: {num_users}, Total movies: {num_movies}")
    
    # Parameters
    num_permutations = 100
    num_bands = 20
    threshold = 0.5
    num_workers = cpu_count()

    permutations = generate_permutations(num_permutations, num_movies, seed)
    
    signature_matrix = minhash_signature_with_permutations_parallel(user_movie_matrix, permutations, num_workers)
    print(f"Signature matrix: {signature_matrix.shape}")
    
    candidate_pairs = lsh_with_hashing(signature_matrix, num_bands, threshold)
    print(f"Total number of candidate pairs: {len(candidate_pairs)}")

    # Perform secondary checks in parallel
    similar_users = filter_similar_users(candidate_pairs, signature_matrix, threshold, num_workers)
    print(f"Total number of similar pairs: {len(similar_users)}")

    lsh_similarity_scores = []
    for u1, u2 in similar_users:
        sim = jaccard_similarity(u1, u2, user_movie_matrix)
        if sim > threshold:
            lsh_similarity_scores.append((u1, u2, sim))

    print(f"Total number of final similar pairs: {len(lsh_similarity_scores)}")

    # Sort the pairs by similarity score
    lsh_similarity_scores = sorted(lsh_similarity_scores, key=lambda x: x[2], reverse=False)

    plot_similarity_scores(lsh_similarity_scores)

    # Write to output
    with open(output_file, "w") as f:
        # Write headers
        f.write("user1,user2,similarity_score\n")

        for u1, u2, lsh_score in lsh_similarity_scores:
            f.write(f"{u1},{u2},{lsh_score:.4f}\n")