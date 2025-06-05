import hashlib

import numpy as np
import cupy as cp

from rescue_cpu import *
from rescue_gpu import *
from sha3 import *

class MerkleTree:
    def __init__(self, data, hash_func="Rescue"):
        self.hash_func = hash_func
        self.data = data
        self.tree = []

        self.leaf_hashes = self._hash_leaves(data)
        self.root = self._build_tree(self.leaf_hashes)

    def _hash_leaves(self, data):
        # GPU batch hash
        ans = []

        if self.hash_func == "Rescue":
            for x in self.data:
                hash_val = rescue_hash_cpu([x])[0]
                ans.append(hash_val)
        elif self.hash_func == "SHA3":
            for x in self.data:
                hash_val = sha3_keccak_cpu(x)
                ans.append(hash_val)
        return ans

    def _build_tree(self, hashes):
        current = hashes
        while len(current) > 1:
            if len(current) % 2 == 1:
                current = np.concatenate([current, current[-1:]])
            self.tree.append(current)
            left_sub = current[::2]
            right_sub = current[1::2]

            if self.hash_func == "Rescue":
                combined = [(left << 32) ^ right for left, right in zip(left_sub, right_sub)]
                current = [rescue_hash_cpu([combine])[0] for combine in combined]
            elif self.hash_func == "SHA3":
                combined = [(left << 32) ^ right for left, right in zip(left_sub, right_sub)]
                current = [sha3_keccak_cpu(combine) for combine in combined]
        self.tree.append(current)
        return current[0]

    def get_root_value(self):
        return self.root

    def proof_of_inclusion(self, index):
        proof = []
        current_level = self.tree[0]
        # Sibling index calculation
        i = 0
        sibling_index = index

        while len(current_level) > 1:
            sibling_index = index ^ 1 # Get the sibling index
            proof.append((self.tree[i][sibling_index], sibling_index % 2))
            index //= 2
            i += 1
            current_level = self.tree[i]

        return proof

    def verify_proof(self, leaf, proof):
        current = leaf

        for sibling in proof:
            sibling_val = sibling[0]
            
            if sibling[1] == 1:  # Left sibling
                current = (current << 32) ^ sibling_val
            else:
                current = (sibling_val << 32) ^ current

            if self.hash_func == "Rescue":
                current = rescue_hash_cpu([current])[0]
            elif self.hash_func == "SHA3":
                current = sha3_keccak_cpu(current)
        return current == self.root

class GPUMerkleTree:
    def __init__(self, data, hash_func="Rescue"):
        self.hash_func = hash_func
        self.data = data
        self.tree = []

        self.leaf_hashes = self._hash_leaves(data)
        print("Leaf hashes:", self.leaf_hashes.shape)
        self.root = self._build_tree_gpu(self.leaf_hashes)
        

    def _hash_leaves(self, data):
        # GPU batch hash
        if self.hash_func == "Rescue":
            return batch_rescue_hash_gpu(self.data)[:, 0]
        elif self.hash_func == "SHA3":
            return sha3_keccak_gpu(np.array(self.data))
        else:
            raise ValueError("Unknown hash function")

    def _build_tree_gpu(self, hashes):
        current = hashes
        while current.shape[0] > 1:
            if current.shape[0] % 2 == 1:
                if self.hash_func == "Rescue":
                    current = cp.concatenate([current, current[-1:]])
                elif self.hash_func == "SHA3":
                    current = np.concatenate([current, current[-1:]])
            self.tree.append(current)
            left = current[::2]
            right = current[1::2]

            if self.hash_func == "Rescue":
                combined = (left.astype(cp.uint64) << 32) ^ right.astype(cp.uint64)
                current = batch_rescue_hash_gpu(cp.array(combined))[:, 0]
            elif self.hash_func == "SHA3":
                combined = (left.astype(np.uint64) << 32) ^ right.astype(np.uint64)
                current = sha3_keccak_gpu(combined)
        self.tree.append(current)
        return current[0]

    def get_root_value(self):
        if self.hash_func == "Rescue":
            return self.root.get()
        elif self.hash_func == "SHA3":
            return self.root

    def proof_of_inclusion(self, index):
        proof = []
        current_level = self.tree[0]
        # Sibling index calculation
        i = 0
        sibling_index = index

        while current_level.shape[0] > 1:
            sibling_index = index ^ 1 # Get the sibling index
            proof.append((self.tree[i][sibling_index], sibling_index % 2))
            index //= 2
            i += 1
            current_level = self.tree[i]

        return proof

    def verify_proof(self, leaf, proof):
        current = leaf
        if self.hash_func == "Rescue":
            current = cp.array([current], dtype=cp.uint64)

        for sibling in proof:
            sibling_val = sibling[0]
            
            if sibling[1] == 1:  # Left sibling
                current = (current.astype(np.uint64) << 32) ^ sibling_val.astype(np.uint64)
            else:
                current = (sibling_val.astype(np.uint64) << 32) ^ current.astype(np.uint64)

            if self.hash_func == "Rescue":
                current = batch_rescue_hash_gpu(cp.array(current, dtype=cp.uint64))[:, 0]
            elif self.hash_func == "SHA3":
                current = sha3_keccak_gpu(np.array([current]))[0]
        if self.hash_func == "Rescue":
            current = current[0]
        return current == self.root

if __name__ == "__main__":
    # Example usage of the MerkleTree class
    data = np.random.randint(0, 100, size=10000, dtype=np.int64)

    merkle_tree = MerkleTree(data)
    print("Merkle Root:", merkle_tree.root.value)
