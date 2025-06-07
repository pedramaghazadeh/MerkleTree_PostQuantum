import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

# --- CUDA Kernel for Keccak-f[1600] ---
cuda_code = """
#include <stdint.h>

#define ROL64(a, offset) (((a) << offset) | ((a) >> (64 - offset)))

__device__ __constant__ uint64_t keccakf_rndc[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL,
  0x800000000000808aULL, 0x8000000080008000ULL,
  0x000000000000808bULL, 0x0000000080000001ULL,
  0x8000000080008081ULL, 0x8000000000008009ULL,
  0x000000000000008aULL, 0x0000000000000088ULL,
  0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL,
  0x8000000000008089ULL, 0x8000000000008003ULL,
  0x8000000000008002ULL, 0x8000000000000080ULL,
  0x000000000000800aULL, 0x800000008000000aULL,
  0x8000000080008081ULL, 0x8000000000008080ULL,
  0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __constant__ int keccakf_rotc[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

__device__ __constant__ int keccakf_piln[25] = {
     0,  6, 12, 18, 24,
     3,  9, 10, 16, 22,
     1,  7, 13, 19, 20,
     4,  5, 11, 17, 23,
     2,  8, 14, 15, 21
};

__device__ void keccakf(uint64_t *st) {
    int i, j, round;
    uint64_t t, bc[5];

    for (round = 0; round < 24; round++) {
        for (i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROL64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        t = st[1];
        for (i = 0; i < 24; i++) {
            j = keccakf_piln[i];
            bc[0] = st[j];
            st[j] = ROL64(t, keccakf_rotc[i]);
            t = bc[0];
        }

        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++) bc[i] = st[j + i];
            for (i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        st[0] ^= keccakf_rndc[round];
    }
}

extern "C" __global__ void keccak256_hash(uint64_t *input, uint64_t *output, int num_inputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_inputs) return;

    uint64_t state[25] = {0};
    state[0] = input[idx]; // Absorb input

    keccakf(state);       // Permute state

    output[idx] = state[0]; // Squeeze output (first 64 bits of digest)
}
"""

mod = SourceModule(cuda_code)
keccak_hash_gpu = mod.get_function("keccak256_hash")

# --- GPU Wrapper ---
def sha3_keccak_gpu(inputs):
    inputs = np.asarray(inputs, dtype=np.uint64)
    output = np.zeros_like(inputs, dtype=np.uint64)

    input_gpu = cuda.mem_alloc(inputs.nbytes)
    output_gpu = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(input_gpu, inputs)

    block_size = 256
    grid_size = (len(inputs) + block_size - 1) // block_size
    keccak_hash_gpu(
        input_gpu, output_gpu, np.int32(len(inputs)),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )

    cuda.memcpy_dtoh(output, output_gpu)
    return output

# --- CPU Keccak-f[1600] (GPU-aligned) ---
KECCAKF_ROTC = np.array([
    0,  1, 62, 28, 27,
   36, 44,  6, 55, 20,
    3, 10, 43, 25, 39,
   41, 45, 15, 21,  8,
   18,  2, 61, 56, 14
], dtype=np.int32)

KECCAKF_PILN = np.array([
     0,  6, 12, 18, 24,
     3,  9, 10, 16, 22,
     1,  7, 13, 19, 20,
     4,  5, 11, 17, 23,
     2,  8, 14, 15, 21
], dtype=np.int32)

KECCAKF_RNDC = np.array([
  0x0000000000000001, 0x0000000000008082,
  0x800000000000808a, 0x8000000080008000,
  0x000000000000808b, 0x0000000080000001,
  0x8000000080008081, 0x8000000000008009,
  0x000000000000008a, 0x0000000000000088,
  0x0000000080008009, 0x000000008000000a,
  0x000000008000808b, 0x800000000000008b,
  0x8000000000008089, 0x8000000000008003,
  0x8000000000008002, 0x8000000000000080,
  0x000000000000800a, 0x800000008000000a,
  0x8000000080008081, 0x8000000000008080,
  0x0000000080000001, 0x8000000080008008
], dtype=np.uint64)

def rol64(x, n):
    x = np.uint64(x)
    n = int(n)
    return np.uint64(((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF)


def keccakf_cpu(state):
    for round in range(24):
        bc = np.zeros(5, dtype=np.uint64)
        for i in range(5):
            bc[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20]
        for i in range(5):
            t = bc[(i + 4) % 5] ^ rol64(bc[(i + 1) % 5], 1)
            for j in range(0, 25, 5):
                state[j + i] ^= t

        t = state[1]
        for i in range(24):
            j = KECCAKF_PILN[i]
            bc[0] = state[j]
            state[j] = rol64(t, KECCAKF_ROTC[i])
            t = bc[0]

        for j in range(0, 25, 5):
            for i in range(5):
                bc[i] = state[j + i]
            for i in range(5):
                state[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5]

        state[0] ^= KECCAKF_RNDC[round]
    return state

def sha3_keccak_cpu(x):
    state = np.zeros(25, dtype=np.uint64)
    state[0] = x
    state = keccakf_cpu(state)
    return state[0]

# --- Merkle Tree Construction ---
def build_merkle_tree(hashed_leaves, hash_func):
    tree = [hashed_leaves]
    current = hashed_leaves
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current), 2):
            l = current[i]
            r = current[i + 1] if i + 1 < len(current) else l
            combined = np.array([l ^ r], dtype=np.uint64)
            hashed = hash_func(combined)[0]
            next_level.append(hashed)
        current = np.array(next_level, dtype=np.uint64)
        tree.append(current)
    return tree

def get_merkle_root(tree):
    return tree[-1][0]

def generate_proof(index, tree):
    proof = []
    for level in tree[:-1]:
        sibling = index ^ 1
        if sibling < len(level):
            proof.append(level[sibling])
        index //= 2
    return proof

def verify_proof(leaf, proof, root, index, hash_func):
    current = leaf
    for sibling in proof:
        combined = np.array([current ^ sibling], dtype=np.uint64)
        current = hash_func(combined)[0]
        index //= 2
    return current == root

# --- Benchmarking ---
if __name__ == "__main__":
    leaves = np.arange(1000, dtype=np.uint64)

    # CPU Benchmark (GPU-aligned version)
    start = time.time()
    cpu_hashes = np.array([sha3_keccak_cpu(x) for x in leaves], dtype=np.uint64)
    cpu_time = time.time() - start
    print("CPU Time:", cpu_time)

    # GPU Benchmark
    start = time.time()
    gpu_hashes = sha3_keccak_gpu(leaves)
    gpu_time = time.time() - start
    print("GPU Time:", gpu_time)

    # --- Compare CPU and GPU Hash Outputs ---
    print("\nComparing CPU and GPU Hashes (first 10 entries):")
    for i in range(10):
        print(f"Index {i}: CPU = {hex(cpu_hashes[i])}, GPU = {hex(gpu_hashes[i])}, Match = {cpu_hashes[i] == gpu_hashes[i]}")

    # Small Merkle Tree Example
    test_leaves = np.array([i + 1000 for i in range(8)], dtype=np.uint64)
    hashed_leaves = sha3_keccak_gpu(test_leaves)
    tree = build_merkle_tree(hashed_leaves, sha3_keccak_gpu)
    root = get_merkle_root(tree)
    print("Merkle Root:", hex(root))

    # Proof verification
    index = 4
    leaf = hashed_leaves[index]
    proof = generate_proof(index, tree)
    valid = verify_proof(leaf, proof, root, index, sha3_keccak_gpu)
    print(f"Merkle Proof valid (GPU): {valid}")
