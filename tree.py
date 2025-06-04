import hashlib

import numpy as np
import cupy as cp

from rescue_compare import *
from sha3 import *

class Node():
    def __init__(self, value=None, left=None, right=None, content=None, hash_function=None):
        self.value = value
        self.content = content
        self.left = left
        self.right = right
        self.is_copied = False
        self.hash_func = hash_function if hash_function else "SHA256"

        # If content is provided, compute the hash of the content
        if content is not None:
            self.value = self.hash(content)

    def hash(self, val):
        if self.hash_func == "SHA256":
            return hashlib.sha256(val).hexdigest()
        elif self.hash_func == "SHA3":
            # Use CPU version of SHA3 hash
            return sha3_keccak_cpu(val)
        elif self.hash_func == "Rescue":
            # Use CPU version of Rescue hash
            return rescue_hash_cpu(val)[0]
    def __str__(self):
        return f"Node(value={self.value})"

    def copy(self):
        new_node = Node(self.value, self.left, self.right, self.content, self.hash_func)
        new_node.is_copied = True
        return new_node


class MerkleTree():
    def __init__(self, data: list[str], hash_function=None):
        self.hash_func = hash_function if hash_function else "SHA256"
        # Building the tree
        self.root = self.build_tree(data)
        

    def build_tree(self, data):
        leaves = [Node(value=d, content=d, hash_function=self.hash_func) for d in data]
        if len(leaves) == 0:
            raise ValueError("Data must contain at least one element.")
        elif len(leaves) == 1:
            root = leaves[0]
        else:
            root = self.recursive_build(leaves)
            print("Merkle Tree built successfully.")
        return root

    def recursive_build(self, data):
        if len(data) == 2:
            left = Node(content=data[0].content, hash_function=self.hash_func)
            right = Node(content=data[1].content, hash_function=self.hash_func)
            return Node(content=left.value + right.value, left=left, right=right, hash_function=self.hash_func)

        if len(data) % 2 == 1:
            data.append(data[-1].copy()) # This way we enforce all leaves to be at the same level and round up the number of leaves to the nearest power of two

        mid = len(data) // 2
        left = self.recursive_build(data[:mid])
        right = self.recursive_build(data[mid:])
        return Node(content=((left.value << 32) ^ right.value), left=left, right=right, hash_function=self.hash_func)

    def print_tree(self):
        if self.root:
            self.print_subtree(self.root)
        else:
            print("The tree is empty.")

    def print_subtree(self, node):
        print(f"Node value: {node.value} and content: {node.content}")
        if node.left:
            print("Left child:")
            self.print_subtree(node.left)
        if node.right:
            print("Right child:")
            self.print_subtree(node.right)

    def get_root_value(self):
        return self.root.value

class GPUMerkleTree:
    def __init__(self, data, hash_func="Rescue"):
        self.hash_func = hash_func
        self.data = data
        self.tree = []

        self.leaf_hashes = self._hash_leaves(data)
        self.root = self._build_tree_gpu(self.leaf_hashes)
        

    def _hash_leaves(self, data):
        # GPU batch hash
        if self.hash_func == "Rescue":
            return rescue_hash_gpu(self.data)
        elif self.hash_func == "SHA3":
            return sha3_keccak_gpu(np.array(self.data))
        else:
            raise ValueError("Unknown hash function")

    def _build_tree_gpu(self, hashes):
        current = hashes
        while current.shape[0] > 1:
            # print(f"Current level size: {current.shape[0]}")
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
                current = rescue_hash_gpu(combined)
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

        for sibling in proof:
            sibling_val = sibling[0]
            
            if sibling[1] == 1:  # Left sibling
                current = (current.astype(np.uint64) << 32) ^ sibling_val.astype(np.uint64)
            else:
                current = (sibling_val.astype(np.uint64) << 32) ^ current.astype(np.uint64)

            if self.hash_func == "Rescue":
                current = rescue_hash_gpu(cp.array([current]))[0]
            elif self.hash_func == "SHA3":
                current = sha3_keccak_gpu(np.array([current]))[0]
        return current == self.root

if __name__ == "__main__":
    # Example usage of the MerkleTree class
    data = np.random.randint(0, 100, size=10000, dtype=np.int64)

    merkle_tree = MerkleTree(data)
    print("Merkle Root:", merkle_tree.root.value)
