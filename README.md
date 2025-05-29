# Merkle Tree Construction with Post-Quantum Secure Hash Functions: SHA-3 and Rescue
## ECE 268 Security of Hardware Embedded Systems Final Project

### Basics

A CPU/GPU implemnetation of Merkle Tree with SHA-3 and Rescue hash functions (from scratch) with a verifier for the hashing process. Merkle Tree is popular in hashing distributed systems and/or databases.

### Merkle Tree
A Merkle tree is a binary tree in which each leaf node contains a hash of a data block, and each internal node contains the hash of the concatenation of its children's hashes. The tree culminates in a single hash called the Merkle root, which provides a compact representation of the entire data set.

Given a set of $n$ data blocks $D_1, D_2, \dots, D_n$, the corresponding leaves are $L_i = H(D_i)$, where $H$ is a cryptographic hash function. Internal nodes are computed recursively as $P_i = H(L_{2i} \| L_{2i+1})$, where $\|$ denotes concatenation. Merkle proofs allow verification of a leaf's inclusion using a logarithmic number of hashes, providing scalability and efficiency.

![Hash_Tree svg](https://github.com/user-attachments/assets/eedfe647-6401-4768-b775-49391a742acb)

### SHA-3 hash algorithm
SHA-3, based on the Keccak algorithm, is a sponge-based cryptographic hash function standardized by NIST in 2015. It operates on a fixed-size internal state (1600 bits) divided into a bitrate $r$ and a capacity $c$ such that $r + c = 1600$. The sponge construction consists of two phases: absorbing and squeezing. During absorbing, message blocks are XORed into the state, followed by permutation. After all input is absorbed, the squeezing phase extracts the output hash.

The core permutation function, Keccak-f[1600], consists of 24 rounds, each applying a sequence of transformations (\textit{theta}, \textit{rho}, \textit{pi}, \textit{chi}, and \textit{iota}) to the state array organized as a $5 \times 5 \times 64$ cube. These transformations ensure diffusion and non-linearity, providing resistance against collision and preimage attacks.

SHA-3's post-quantum resistance stems from its output length: for a $n$-bit output, Grover’s algorithm provides only a $2^{n/2}$ security level, making SHA-3-256 suitable for 128-bit post-quantum security.

![SpongeConstruction svg](https://github.com/user-attachments/assets/2335c8cc-1311-4bc1-9119-810c44001832)

### Rescue hash algorithm
Rescue is a cryptographic hash function designed specifically for zero-knowledge proof systems and post-quantum resistance. Unlike traditional bit-based constructions, Rescue operates over finite fields and is optimized for algebraic structures used in zk-SNARKs and other proof systems.

Rescue uses an SPN (Substitution-Permutation Network) structure composed of multiple rounds. Each round includes nonlinear S-boxes applied to state elements (typically raising each element to a power $\alpha$ and $\alpha^{-1}$), addition of round constants, and multiplication by an MDS (Maximum Distance Separable) matrix to ensure diffusion.

For this project, Rescue is implemented over a prime field $\mathbb{F}_p$ where $p = 2^{64} - 59$. This choice allows efficient modular arithmetic on standard 64-bit processors and simplifies GPU parallelization using libraries like CuPy.

![rescue-prime-sponge](https://github.com/user-attachments/assets/bdb81e6c-0e8a-4799-b7cc-08471958c881)

### Requirements
Necessary packages can be installed via
```
pip install -r requirements.txt
```

This project was developed on Cuda 12 drivers and uses CuPy for the GPU implementation. The details regarding the miniconda environment can be found in conda.yaml file.

### Running the repository
```
python main.py --data <List of strings to include in the Merkle Tree> --hash-function <SHA256, SHA3, Rescue> --print-tree --device <cpu, gpu>
```
