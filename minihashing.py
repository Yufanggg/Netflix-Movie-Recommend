# all necessary packages
import numpy as np
import hashlib
import random

class NetflixSimiarlity:
    def __init__(self, fileName):
        self.fileName = fileName
        self.loaded_data = np.load(self.fileName)


    # Compute a hash value for an element using hashlib with a random seed.
    def hashlib_hash_function(self, element, seed):  
        # Combine the seed with the element to ensure randomness
        combined = f"{seed}-{element}".encode('utf-8')
        hash_object = hashlib.sha256(combined)
        return int(hash_object.hexdigest(), 16)  # Convert hash to an integer
    

    # Compute minihash signatures
    def compute_minhash_signature(self, row, num_hashes):
        seeds = [random.randint(0, 2**32-1) for _ in range(num_hashes)]
        signature = []

        # for each hash function, compute the minimum hash value for the row
        for seed in seeds:
            min_hash = float("inf")
            for element in row:
                h = self.hashlib_hash_function(element=element, seed=seed)
                min_hash = min(min_hash, h)
            signature.append(min_hash)

        return signature
    
    # Process the Entire dataset
    def process_data(self, num_hashes=10, max_value=1000):
        # Ensure each row is a set of strings
        if not isinstance(self.loaded_data[0][0], str):
            data = [[str(element) for element in row] for row in self.loaded_data]

        # Compute MinHash signatures for all rows
        signatures = []
        for row in data:
            signature = self.compute_minhash_signature(row=row, num_hashes=num_hashes)
            signatures.append(signature)

        return signatures


