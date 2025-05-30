import hashlib

import numpy as np
import cupy as cp

from rescue_compare import *
from sha3 import *

class Node():
    def __init__(self, value=None, left=None, right=None, content=None, hash_function=None, device="cpu"):
        self.value = value
        self.content = content
        self.left = left
        self.right = right
        self.is_copied = False
        self.hash_func = hash_function if hash_function else "SHA256"
        self.device = device

        # If content is provided, compute the hash of the content
        if content is not None:
            self.value = self.hash(content)

    def hash(self, val):
        if self.hash_func == "SHA256":
            # print(type(hashlib.sha256(val.encode('utf-8')).hexdigest()), hashlib.sha256(val.encode('utf-8')).hexdigest(), "SHA256 Hash")
            return hashlib.sha256(val.encode('utf-8')).hexdigest()
        if self.hash_func == "SHA3":
            if self.device == "cpu":
                # Use CPU version of SHA3 hash
                # print(val, "Value for SHA3 Hash CPU")
                string_bytes = val.encode('utf-8')
                int_val = int.from_bytes(string_bytes[:8], byteorder='big')
                return hex(sha3_keccak_cpu(int_val))[2:]
            else:
                # Use GPU version of SHA3 hash
                string_bytes = val.encode('utf-8')
                int_val = int.from_bytes(string_bytes[:8], byteorder='big')
                return hex(sha3_keccak_gpu(int_val)[0])[2:]

        if self.hash_func == "Rescue":
            if self.device == "cpu":
                # Use CPU version of Rescue hash
                string_bytes = val.encode('utf-8')
                int_val = int.from_bytes(string_bytes[:8], byteorder='big')
                # print(int_val, "Integer value for Rescue Hash CPU")
                # Compute the Rescue hash using CPU
                return hex(rescue_hash_cpu(int_val)[0])[2:]  # Convert to hex and remove '0x' prefix
            else:
                # Use GPU version of Rescue hash
                string_bytes = val.encode('utf-8')
                int_val = int.from_bytes(string_bytes[:8], byteorder='big')
                # print(int_val, "Integer value for Rescue Hash GPU")
                # Compute the Rescue hash using GPU
                return hex(int(rescue_hash_gpu(int_val)[0]))[2:]


    def __str__(self):
        return f"Node(value={self.value})"

    def copy(self):
        new_node = Node(self.value, self.left, self.right, self.content, self.hash_func)
        new_node.is_copied = True
        return new_node


class MerkleTree():
    def __init__(self, data: list[str], hash_function=None, device="cpu"):
        self.hash_func = hash_function if hash_function else "SHA256"
        self.device = device
        # Building the tree
        self.root = self.build_tree(data)
        

    def build_tree(self, data):
        leaves = [Node(value=d, content=d, hash_function=self.hash_func, device=self.device) for d in data]
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
            left = Node(content=data[0].content, hash_function=self.hash_func, device=self.device)
            right = Node(content=data[1].content, hash_function=self.hash_func, device=self.device)
            return Node(content=left.value + right.value, left=left, right=right, hash_function=self.hash_func, device=self.device)

        if len(data) % 2 == 1:
            data.append(data[-1].copy()) # This way we enforce all leaves to be at the same level and round up the number of leaves to the nearest power of two

        mid = len(data) // 2
        left = self.recursive_build(data[:mid])
        right = self.recursive_build(data[mid:])
        return Node(content=left.value + right.value, left=left, right=right, hash_function=self.hash_func, device=self.device)

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

if __name__ == "__main__":
    # Example usage of the MerkleTree class
    data = ["a", "b", "c", "de", "f"]

    merkle_tree = MerkleTree(data)
    print("Merkle Root:", merkle_tree.root.value)
