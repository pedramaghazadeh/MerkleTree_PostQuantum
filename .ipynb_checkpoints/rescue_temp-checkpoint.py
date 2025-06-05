# Imports
import cupy as cp
import numpy as np
import time

# Prime for finite field
p = 2**31 - 1  # 2147483647

# S-box exponent
alpha = 5

# Inverse S-box exponent (mod p - 1)
inv_alpha = pow(alpha, -1, p - 1)

# Number of rescue rounds
num_rounds = 10

# State size = input length + capacity
state_size = 3

# Number of elements in output
rate = 2

# MDS matrix
mds_matrix = cp.array([
    [2, 3, 1],
    [1, 1, 4],
    [3, 5, 6]
], dtype=cp.int64)

# Round constants
round_constants = [np.array([(i * j + 1) % p for j in range(state_size)], dtype=object) for i in range(2 * num_rounds)]
round_constants = [cp.array(rc, dtype=cp.int64) for rc in round_constants]

# S-Box function
def sbox(curr_state):
    return pow(int(curr_state), alpha, p)

# Inverse S-Box function
def inv_s_box(curr_state):
    return pow(int(curr_state), inv_alpha, p)

# MDS matrix multiplication
def mds_multiply(state):
    return cp.mod(cp.dot(mds_matrix, state), p)

# Rescue hash function
def rescue_hash(inputs):
    if not isinstance(inputs, list):
        inputs = [inputs]  # Ensure inputs is a list
    # Pad inputs: add single 1, then 0s
    padded = inputs + [1] + [0] * (state_size - len(inputs) - 1)
    # Initialize state as CuPy array
    state = cp.array(padded, dtype=cp.int64)

    # Apply functions through rounds
    for i in range(num_rounds):
        # S-box layer
        state_np = cp.asnumpy(state)
        state = cp.array([sbox(curr_state) for curr_state in state_np], dtype=cp.int64)
        # MDS matrix multiplication #1
        state = mds_multiply(state)
        # Add round constant #1
        state = cp.mod(state + round_constants[2 * i], p)
        # Inverse S-box layer
        state_np = cp.asnumpy(state)
        state = cp.array([inv_s_box(curr_state) for curr_state in state_np], dtype=cp.int64)
        # MDS matrix multiplication #2
        state = mds_multiply(state)
        # Add round constant #2
        state = cp.mod(state + round_constants[2 * i + 1], p)
    # Return first rate elements of final state as Python list
    return cp.asnumpy(state[:rate]).tolist()

# Test cases
if __name__ == "__main__":
    print(cp.__version__)
    print(cp.cuda.runtime.getDeviceCount())
    
    # Empty input
    start_time = time.perf_counter()
    empty_test = rescue_hash([])
    end_time = time.perf_counter()
    print("Rescue hash of []:", empty_test)
    print(f"Time: {(end_time - start_time) * 1000:.3f} ms")
    
    # Non-empty input
    start_time = time.perf_counter()
    non_empty_test = rescue_hash([16, 234])
    end_time = time.perf_counter()
    print("Rescue hash of [16, 234]:", non_empty_test)
    print(f"Time: {(end_time - start_time) * 1000:.3f} ms")
    
    # Additional input
    start_time = time.perf_counter()
    additional_test = rescue_hash([100, 200])
    end_time = time.perf_counter()
    print("Rescue hash of [100, 200]:", additional_test)
    print(f"Time: {(end_time - start_time) * 1000:.3f} ms")
    
    # Ensure hash function is deterministic
    start_time = time.perf_counter()
    is_deterministic = rescue_hash([45, 125]) == rescue_hash([45, 125])
    end_time = time.perf_counter()
    print("Deterministic" if is_deterministic else "Not deterministic")
    print(f"Time: {(end_time - start_time) * 1000:.3f} ms")