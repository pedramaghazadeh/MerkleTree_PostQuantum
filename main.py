import argparse

from tree import MerkleTree
from sha3 import *
from rescue import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merkle Tree Implementation")
    parser.add_argument("--data", nargs="+", help="List of strings to include in the Merkle Tree")
    parser.add_argument("--hash_function", choices=["SHA256", "SHA3", "Rescue"], default="SHA256", help="Hash function to use")
    parser.add_argument("--print_tree", action="store_true", help="Print the structure of the Merkle Tree")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Device to use for computation")
    args = parser.parse_args()

    if args.data:
        merkle_tree = MerkleTree(data=args.data, hash_function=args.hash_function)
        print("Merkle Root:", merkle_tree.get_root_value())
    else:
        print("No data provided.")

    if args.print_tree:
        merkle_tree.print_tree()

    print("Using device:", args.device)