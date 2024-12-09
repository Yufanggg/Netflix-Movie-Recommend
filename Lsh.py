
from scipy.integrate import quad as integrate
import string
import random
import struct


from Storage import DictSetStorage, DictListStorage


def _random_name(length):
    return ''.join(random.choice(string.ascii_lowercase)
                   for _ in range(length)).encode('utf8')


def _false_positive_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, err = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, err = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight,
                   false_negative_weight):
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r, fp, fn)
    return opt


class MinHashLSH(object):

    def __init__(self, threshold=0.5, d=128, weights=(0.5, 0.5),
                 params=None, storage_config=None):
        if storage_config is None:
            storage_config = {'type': 'dict'}

        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = d
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > d:
                raise ValueError("The product of b and r in params is "
                                 "{} * {} = {} -- it must be less than d {}. ".format(self.b, self.r, self.b * self.r,
                                                                                      d))
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r, self.fp, self.fn = _optimal_param(threshold, d, false_positive_weight,
                                                              false_negative_weight)
            print('the best parameter b={},r={},fp={},fn={}'.format(self.b, self.r, self.fp, self.fn))

        basename = storage_config.get('basename', _random_name(11))
        # 哈希表，也就是b个hash表
        self.hashtables = []
        # 这个就是分段，也就是r段数据
        self.hashranges = []
        for i in range(self.b):
            name = b''.join([basename, b'_bucket_', struct.pack('>H', i)])
            item = DictSetStorage(storage_config, name=name)
            self.hashtables.append(item)

            self.hashranges.append((i * self.r, (i + 1) * self.r))

        self.keys = DictListStorage(storage_config, name=b''.join([basename, b'_keys']))

    def insert(self, key, minhash):
        self._insert(key, minhash, buffer=False)

    def _insert(self, key, minhash, buffer=False):
        if key in self.keys:
            raise ValueError("key already exists")
        Hs = []
        for start, end in self.hashranges:
            Hs.append(self._H(minhash.hashvalues[start:end]))
        #在另外的表种存储key和hash value表（比如分成了b段那就有b个hashvalue,每个value在不同的hash表种）
        self.keys.insert(key, *Hs, buffer=buffer)
        # 更新每个桶（hash表）中的key
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, key, buffer=buffer)


    # 查询的逻辑，检查每个表的筒里面，有没有和被查询的 hashvalue一样的在筒中的value，如果有，通过hashvalue找到key
    def query(self, minhash):
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H):
                candidates.add(key)

        return list(candidates)

    def _H(self, hs):
        return bytes(hs.byteswap().data)