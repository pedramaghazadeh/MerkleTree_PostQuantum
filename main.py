import argparse
import time

import numpy as np
import cupy as cp

from tree import MerkleTree, GPUMerkleTree

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merkle Tree Implementation")
    parser.add_argument("--data-length", type=int, default=1_000, help="Length of the input data")
    parser.add_argument("--hash-function", choices=["SHA256", "SHA3", "Rescue"], default="SHA256", help="Hash function to use")
    # parser.add_argument("--print-tree", action="store_true", help="Print the structure of the Merkle Tree")
    args = parser.parse_args()

    print("Input Data Length:", args.data_length)

    test_input = np.random.randint(0, 1_000_000_000, size=args.data_length, dtype=np.int64)
    test_input_gpu = cp.array(test_input, dtype=cp.uint64)

    print("\n\nUsing GPU for Merkle Tree construction.")
    st = time.time()
    # Initialize the Merkle Tree with GPU support
    if args.hash_function == "SHA3":
        test_input = np.random.randint(0, 1_000_000_000, size=args.data_length, dtype=np.uint64)
        test_input_gpu = test_input

    merkle_tree = GPUMerkleTree(data=test_input_gpu, hash_func=args.hash_function)
    en = time.time()
    time_gpu = en - st
    print(f"Time taken (GPU): {en - st}s")
    print("Merkle Root:", merkle_tree.get_root_value(), "\n\n")
    # Proof generation and verification
    index = np.random.randint(0, args.data_length)

    leaf = merkle_tree.leaf_hashes[index]
    proof = merkle_tree.proof_of_inclusion(index)
    valid = merkle_tree.verify_proof(leaf, proof)
    print(f"Merkle Proof validity (GPU) for index {index}: {valid}")
    print("=" * 50, "\n")

    

    st = time.time()
    print("Using CPU for Merkle Tree construction.")
    merkle_tree = MerkleTree(data=test_input, hash_function=args.hash_function)
    en = time.time()
    time_cpu = en - st
    print(f"Time taken (CPU): {en - st}s")
    print("Merkle Root:", merkle_tree.get_root_value())

    print("\n\nExecution Time Comparison")
    print(f"CPU Time: {time_cpu:.4f} seconds")
    print(f"GPU Time: {time_gpu:.4f} seconds")
    print(f"Speedup of GPU over CPU: {time_cpu / time_gpu}x")