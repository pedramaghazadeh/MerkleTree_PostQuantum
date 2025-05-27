### Merkle Tree Construction with Post-Quantum Secure Hash Functions: SHA-3 and Rescue
## ECE 268 Security of Hardware Embedded Systems Final Project

# Basics

A CPU/GPU implemnetation of Merkle Tree with SHA-3 and Rescue hash functions (from scratch) with a verifier for the hashing process. Merkle Tree is popular in hashing distributed systems and/or databases.

# Requirements
Necessary packages can be installed via
```
pip install -r requirements.txt
```

This project was developed on Cuda 12 drivers and uses CuPy for the GPU implementation. The details regarding the miniconda environment can be found in conda.yaml file.

# Running the project
```
python main.py --data <List of strings to include in the Merkle Tree> --hash-function <SHA256, SHA3, Rescue> --print-tree --device <cpu, gpu>
```