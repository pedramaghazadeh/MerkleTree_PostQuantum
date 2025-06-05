import hashlib

import numpy as np
import cupy as cp

from rescue import *
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
            # print(type(hashlib.sha256(val.encode('utf-8')).hexdigest()), hashlib.sha256(val.encode('utf-8')).hexdigest(), "SHA256 Hash")
            return hashlib.sha256(val.encode('utf-8')).hexdigest()
        elif self.hash_func == "SHA3":
            # Use CPU version of SHA3 hash
            # print(val, "Value for SHA3 Hash CPU")
            string_bytes = val.encode('utf-8')
            int_val = int.from_bytes(string_bytes[:8], byteorder='big')
            return hex(sha3_keccak_cpu(int_val))[2:]
        elif self.hash_func == "Rescue":
            # Use CPU version of Rescue hash
            string_bytes = val.encode('utf-8')
            int_val = int.from_bytes(string_bytes[:8], byteorder='big')
            # print(int_val, "Integer value for Rescue Hash CPU")
            # Compute the Rescue hash using CPU
            return hex(rescue_hash_cpu(int_val)[0])[2:]  # Convert to hex and remove '0x' prefix

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
        return Node(content=left.value + right.value, left=left, right=right, hash_function=self.hash_func)

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
    def __init__(self, data: list[str], hash_func="Rescue"):
        self.hash_func = hash_func
        self.data = data
        self.leaf_hashes = self._hash_leaves(data)
        self.root = self._build_tree_gpu(self.leaf_hashes)

    def _hash_leaves(self, data):
        # Convert strings to 64-bit ints
        inputs = []
        for d in data:
            b = d.encode("utf-8")
            val = int.from_bytes(b[:8], 'big')  # truncate to 64-bit
            inputs.append(val)

        # GPU batch hash
        if self.hash_func == "Rescue":
            arr = cp.array(inputs, dtype=cp.uint64)
            return rescue_hash_gpu(arr)
        elif self.hash_func == "SHA3":
            arr = np.array(inputs, dtype=np.uint64)
            return sha3_keccak_gpu(arr)
        else:
            raise ValueError("Unknown hash function")

    def _build_tree_gpu(self, hashes):
        current = hashes
        while current.shape[0] > 1:
            print(f"Current level size: {current.shape[0]}")
            if current.shape[0] % 2 == 1:
                if self.hash_func == "Rescue":
                    current = cp.concatenate([current, current[-1:]])
                elif self.hash_func == "SHA3":
                    current = np.concatenate([current, current[-1:]])

            left = current[::2]
            right = current[1::2]

            combined = (left.astype(cp.uint64) << 32) ^ right.astype(cp.uint64)

            if self.hash_func == "Rescue":
                arr = cp.array(combined, dtype=cp.uint64)
                current = rescue_hash_gpu(arr)
            elif self.hash_func == "SHA3":
                current = sha3_keccak_gpu(combined)
            print("Current level ", current)
        return current[0]

    def get_root_value(self):
        if self.hash_func == "Rescue":
            print(self.root, "Root value in GPU Merkle Tree")
            return hex(int(self.root.get()))[2:]
        elif self.hash_func == "SHA3":
            # Convert the GPU array to a CPU array and then to hex
            return hex(int(self.root))[2:]

if __name__ == "__main__":
    # Example usage of the MerkleTree class
    data = ["a", "b", "c", "de", "f"]

    merkle_tree = MerkleTree(data)
    print("Merkle Root:", merkle_tree.root.value)
