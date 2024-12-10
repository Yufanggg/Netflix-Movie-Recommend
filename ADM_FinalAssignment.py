import numpy as np
import scipy.sparse as sp
import sys
import random
import hashlib
import time
from itertools import combinations
from multiprocessing import Pool, cpu_count, current_process, shared_memory
from math import ceil
import matplotlib.pyplot as plt


class MultiprocessingTaskExecutor:
    def __init__(self, func, num_workers, seed):

        self.func = func
        self.num_workers = num_workers if num_workers else cpu_count()
        self.seed = seed

    def _initialize_worker(self, seed):
        """
        Initialize the random seed for each worker.
        """
        if seed is not None:
            np.random.seed(seed + int(current_process().name.split('-')[-1]))  # Different seed per worker
        else:
            np.random.seed()

    def execute_map(self, tasks):

        with Pool(processes=self.num_workers, initializer=self._initialize_worker, initargs=(self.seed,)) as pool:
            results = pool.map(self.func, tasks)
        return results

    def execute_starmap(self, tasks):

        with Pool(processes=self.num_workers, initializer=self._initialize_worker, initargs=(self.seed,)) as pool:
            results = pool.starmap(self.func, tasks)
        return results


'''## Loading & Preprocessing Data:
Preprocessing involves creating a sparse matrix with binary values, where actual ratings are converted to 1. 
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


'''## Generating random Permutations:
Generating n random permutations of movie indices ensures that each row of the user-movie matrix 
is permuted in a unique and randomized order during the MinHash computation.'''
def generate_permutations(num_permutations, num_movies, seed):

    """Generate random permutations of movie indices."""
    np.random.seed(seed)
    movie_indices = np.arange(num_movies)
    permutations = [np.random.permutation(movie_indices) for _ in range(num_permutations)]

    return permutations



'''## Minhasing with parallel processing'''
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


def minhash_signature_with_permutations_parallel(user_movie_matrix, permutations, num_workers, seed):

    """Generate Minhash signatures in parallel."""
    num_users = user_movie_matrix.shape[0]

    # Prepare user data
    user_data_list = [user_movie_matrix.getrow(user) for user in range(num_users)]

    # Execute in parallel
    executor = MultiprocessingTaskExecutor(func=generate_user_signature, num_workers=num_workers, seed=seed)
    signatures = executor.execute_starmap([(user_data, permutations) for user_data in user_data_list])

    return np.array(signatures).T  # Transpose to get the expected shape





'''## LSH Banding Technique:
We divide the signatures into n bands and split them into chunks and hash before putting them in buckets.
The idea is similar pairs will end up in the same bucket thanks to hashing.
The signatures are divided into n bands, with each band containing a chunk of the signature. 
These chunks are then hashed and placed into buckets. The key idea is that similar pairs are likely to hash into the same bucket, 
facilitating the efficient identification of potential matches.'''
def hash_band(band_signature):
    """Hash a band signature using SHA-256 and return a bucket value."""
    return int(hashlib.sha256(str(band_signature).encode()).hexdigest(), 16)


def lsh_with_hashing(signature_matrix, num_bands, rows_per_band, threshold):
    """Apply LSH with hashing and secondary similarity checks."""
    n_permutations, n_users = signature_matrix.shape
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



'''## Secondary Similarity Check in parallel:
Secondary check is performed to make sure whether the candidate pairs are really similar pairs and possibly get rid of false positives.'''
def create_shared_memory_array(arr):

    """Create a shared memory array from a numpy array."""
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    np.copyto(shared_arr, arr)
    
    return shm, shared_arr


def check_similarity_parallel(task):

    """Wrapper for secondary similarity check."""
    user1, user2, shm_name, shape, dtype, threshold = task

    # Access shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    signature_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    n_permutations = signature_matrix.shape[0]
    user1_signature = signature_matrix[:, user1]
    user2_signature = signature_matrix[:, user2]

    common_signatures = np.sum(user1_signature == user2_signature)
    similarity = common_signatures / n_permutations

    return (user1, user2) if similarity >= threshold else None


def filter_similar_users(candidate_pairs, signature_matrix, threshold, num_workers, seed, batch_size = 1000000):

    """Filter candidate pairs in parallel."""
    
    # Create shared memory for the signature matrix
    shm, shared_signature_matrix = create_shared_memory_array(signature_matrix)
    
    candidate_pairs = list(candidate_pairs)
    num_batches = ceil(len(candidate_pairs) / batch_size)
    similar_users = []

    # Process candidate pairs in batches
    for batch_index in range(num_batches):
        # Create a batch of candidate pairs
        start = batch_index * batch_size
        end = start + batch_size
        batch = candidate_pairs[start:end]

        tasks = [(user1, user2, shm.name, signature_matrix.shape, signature_matrix.dtype, threshold) for user1, user2 in batch]

        # Execute tasks in parallel
        executor = MultiprocessingTaskExecutor(func=check_similarity_parallel, num_workers=num_workers, seed=seed)
        results = executor.execute_map(tasks)

        # Filter None results and append them to the similar_users list
        similar_users.extend([pair for pair in results if pair is not None])

    # Clean up shared memory
    shm.close()
    shm.unlink()

    return similar_users



'''## Jaccard Similarity'''
def create_shared_memory_array_sparse(sparse_matrix):
    
    """Create shared memory for a sparse matrix."""
    shm = shared_memory.SharedMemory(create=True, size=sparse_matrix.data.nbytes)
    shared_data = np.ndarray(sparse_matrix.data.shape, dtype=sparse_matrix.data.dtype, buffer=shm.buf)
    np.copyto(shared_data, sparse_matrix.data)

    return shm, sparse_matrix.indices, sparse_matrix.indptr, sparse_matrix.shape


def jaccard_similarity_task(task):
    
    """Task wrapper for computing Jaccard similarity."""
    u1, u2, shared_name, indices, indptr, shape, threshold = task
    
    # Access shared memory
    shm = shared_memory.SharedMemory(name=shared_name)
    data = np.ndarray((indptr[-1],), dtype=np.float64, buffer=shm.buf)
    
    # Reconstruct sparse matrix
    user_movie_matrix = sp.csr_matrix((data, indices, indptr), shape=shape)

    movies_u1 = user_movie_matrix[u1].indices
    movies_u2 = user_movie_matrix[u2].indices

    intersection = np.intersect1d(movies_u1, movies_u2, assume_unique=True).size
    union = np.union1d(movies_u1, movies_u2).size

    similarity = intersection / union if union != 0 else 0

    return (u1, u2, similarity) if similarity > threshold else None


def compute_jaccard_similarity_parallel(similar_users, user_movie_matrix, threshold, num_workers, batch_size=5000):
    
    """Compute Jaccard similarity for candidate pairs in parallel."""
    
    # Create shared memory
    shm, indices, indptr, shape = create_shared_memory_array_sparse(user_movie_matrix)

    num_batches = ceil(len(similar_users) / batch_size)
    lsh_similarity_scores = []

    for batch_idx in range(num_batches):
        print(f"Processing batch {batch_idx + 1}/{num_batches}...")
        
        # Slice the current batch
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(similar_users))
        batch = similar_users[start_idx:end_idx]

        tasks = [(u1, u2, shm.name, indices, indptr, shape, threshold) for u1, u2 in batch]

        # Use multiprocessing pool
        with Pool(num_workers//2) as pool:
            results = pool.map(jaccard_similarity_task, tasks)

        # Filter out None results and extend to the global result
        lsh_similarity_scores.extend([result for result in results if result is not None])

    return lsh_similarity_scores
    
    
'''## Plot Function'''    
def plot_similarity_scores(similarity_scores):

    # Extract the similarity scores
    scores = [score for _, _, score in similarity_scores]

    # Plot the similarity scores
    plt.figure(figsize=(20, 6))
    plt.plot(scores, marker='o', linestyle='-', color='green')
    plt.title('LSH Similarity Scores of Similar User Pairs')
    plt.xlabel('Pair Index')
    plt.ylabel('Similarity Score')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":

	if len(sys.argv) < 2 or not sys.argv[1].isdigit():
		print("Command-line argument not provided or invalid in Colab.")
		seed = int(input("Enter the random seed: "))
	else:
		seed = int(sys.argv[1])

	start = time.time()
	output_file = "result.txt"
	data_file = "/content/drive/MyDrive/Colab Notebooks/ADVANCES IN DATA MINING/FINAL ASSIGNMENT/Dataset/user_movie_rating.npy"

	# Load data
	user_movie_matrix = load_data(data_file)
	num_users, num_movies = user_movie_matrix.shape
	print(f"Total users: {num_users}, Total movies: {num_movies}")

	# Parameters
	threshold = 0.5
	num_permutations = 100
	num_bands = 16
	rows_per_band = 6
	num_workers = cpu_count()
	print(f"num_workers: {num_workers}, num_permutations:{num_permutations}, num_bands: {num_bands}, rows_per_band: {rows_per_band}")

	permutations = generate_permutations(num_permutations, num_movies, seed)

	signature_matrix = minhash_signature_with_permutations_parallel(user_movie_matrix, permutations, num_workers, seed)
	print(f"Signature matrix: {signature_matrix.shape}")
	print(f"Signature Matrix size: {sys.getsizeof(signature_matrix)} bytes")

	candidate_pairs = lsh_with_hashing(signature_matrix, num_bands, rows_per_band, threshold)
	print(f"Total number of candidate pairs: {len(candidate_pairs)}")

	# Perform secondary checks in parallel
	similar_users = filter_similar_users(candidate_pairs, signature_matrix, threshold, num_workers, seed)
	print(f"Total number of candidate pairs after secondary check: {len(similar_users)}")
	
	lsh_similarity_scores = []
	lsh_similarity_scores = compute_jaccard_similarity_parallel(similar_users, user_movie_matrix, threshold, num_workers)
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

	end = time.time()
	execution_time = (end - start) / 60
	print(f"Execution time: {execution_time}")
