import hashlib
import numpy as np
import struct


def sha1_hash32(data):
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)


class Minhash(object):
    def __init__(self,d=150,seed=42,hashfunc=sha1_hash32,hashvalues=None,permutations=None):
            if hashvalues is not None:
                d=len(hashvalues)
            self.seed=seed
            if not callable(hashfunc):
                raise TypeError('hashfunc must be callable')
            self.hashfunc=hashfunc
            # Initialize hash values
            if hashvalues is not None:
                self.hashvalues = self._parse_hashvalues(hashvalues)
            else:
                self.hashvalues = self._init_hashvalues(d)
            if permutations is not None:
                self.permutations = permutations
            else:
                generator = np.random.RandomState(self.seed)
                self.permutations = np.array([(generator.randint(1, _mersenne_prime, dtype=np.uint64),
                                               generator.randint(0, _mersenne_prime, dtype=np.uint64))
                                              for _ in range(d)], dtype=np.uint64).T
            if len(self) != len(self.permutations[0]):
                raise ValueError("Numbers of hash values and permutations mismatch")
    def _init_hashvalues(self, d):
        return np.ones(d, dtype=np.uint64)*_max_hash

    def _parse_hashvalues(self, hashvalues):
        return np.array(hashvalues, dtype=np.uint64)
    def __len__(self):
        return len(self.hashvalues)

    def update(self, b):

        hv = self.hashfunc(b)
        a, b = self.permutations
        phv = np.bitwise_and((a * hv + b) % _mersenne_prime, np.uint64(_max_hash))
        self.hashvalues = np.minimum(phv, self.hashvalues)
