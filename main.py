import argparse

from tree import MerkleTree, GPUMerkleTree

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merkle Tree Implementation")
    parser.add_argument("--data", nargs="+", help="List of strings to include in the Merkle Tree")
    parser.add_argument("--hash-function", choices=["SHA256", "SHA3", "Rescue"], default="SHA256", help="Hash function to use")
    # parser.add_argument("--print-tree", action="store_true", help="Print the structure of the Merkle Tree")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Device to use for computation")
    args = parser.parse_args()

    print("Input Data:", args.data)

    if args.data:
        if args.device == "gpu":
            print("Using GPU for Merkle Tree construction.")
            merkle_tree = GPUMerkleTree(data=args.data, hash_func=args.hash_function)
            print("Merkle Root:", merkle_tree.get_root_value())
        else:
            print("Using CPU for Merkle Tree construction.")
            merkle_tree = MerkleTree(data=args.data, hash_function=args.hash_function)
            print("Merkle Root:", merkle_tree.get_root_value())
    else:
        print("No data provided.")
