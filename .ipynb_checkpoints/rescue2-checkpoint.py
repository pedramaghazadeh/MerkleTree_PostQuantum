# Imports
import numpy as np
import cupy as cp
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
mds_matrix = np.array([
    [2, 3, 1],
    [1, 1, 4],
    [3, 5, 6]
], dtype=object)

# Round constants
round_constants = [np.array([(i * j + 1) % p for j in range(state_size)], dtype=object) for i in range(2 * num_rounds)]

# S-Box function
def sbox(curr_state):
    return pow(curr_state, alpha, p)

# Inverse S-Box function
def inv_s_box(curr_state):
    return pow(curr_state, inv_alpha, p)

# MDS matrix multiplication
def mds_multiply(state):
    return np.mod(mds_matrix @ state, p)

# Rescue hash function
def rescue_hash(inputs):
    if type(inputs) is not list:
        inputs = [inputs]  # Ensure inputs is a list
    # Pad inputs: add single 1, then 0s
    padded = inputs + [1] + [0] * (state_size - len(inputs) - 1)
    # Update state
    state = np.array(padded, dtype=object)

    # Apply functions through rounds
    for i in range(num_rounds):
        # S-box layer
        state = np.array([sbox(curr_state) for curr_state in state], dtype=object)
        # MDS matrix multiplication #1
        state = mds_multiply(state)
        # Add round constant #1
        state = np.mod(state + round_constants[2 * i], p)
        # Inverse S-box layer
        state = np.array([inv_s_box(curr_state) for curr_state in state], dtype=object)
        # MDS matrix multiplication #2
        state = mds_multiply(state)
        # Add round constant #2
        state = np.mod(state + round_constants[2 * i + 1], p)
    # Return first rate elements of final state
    return list(state[:rate])

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