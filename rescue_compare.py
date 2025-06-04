# Imports
import numpy as np
import cupy as cp
import math
import time

### Initializations
# prime for finite field (same as merkle tree) should be large, like 2**255 - 19
p = 2**60 - 1
# s-box ecurr_stateponent (adds nonlinearity)
alpha = 23
# make sure alpha and p-1 are coprime
if math.gcd(alpha, p - 1) != 1:
    raise ValueError("alpha and p-1 must be coprime!")
# inverse s-box ecurr_stateponent (mod p - 1)
inv_alpha = pow(alpha, -1, p - 1)
# number of rescue rounds
num_rounds = 10
# state size = input length + capacity (e.g. 2 + 1 = 3)
state_size = 3
# number of elements in output
rate = 2

# MDS matrix curr_state -> micurr_state state variables between rounds, introduces diffusion
# both CPU and GPU versions
mds_matrix_cpu = np.array([
    [2, 3, 1],
    [1, 1, 4],
    [3, 5, 6]
], dtype=np.int64)
mds_matrix_gpu = cp.array([
    [2, 3, 1],
    [1, 1, 4],
    [3, 5, 6]
], dtype=cp.uint64)

# round constants (generate unique values)
# both CPU and GPU versions
round_constants_cpu = [np.array([(i * j + 1) % p for j in range(state_size)], dtype=np.int64) for i in range(2 * num_rounds)]
round_constants_gpu = [cp.array([(i * j + 1) % p for j in range(state_size)], dtype=cp.uint64) for i in range(2 * num_rounds)]

##### CPU Functions #####
### S-Box curr_state Function
def s_box_cpu(curr_state):
    # curr_state^alpha mod p
    return pow(int(curr_state), alpha, p)
### Inverse S-Box Function
def inv_s_box_cpu(curr_state):
    # curr_state^alpha_inv mod p
    return pow(int(curr_state), inv_alpha, p)

def mds_multiply_cpu(state):
    # multiply state by MDS matricurr_state
    return np.mod(mds_matrix_cpu @ state, p)

### Rescue Hash Function
def rescue_hash_cpu(inputs):
    if type(inputs) is not list:
        inputs = [inputs] # ensure inputs is a list
    # pad inputs -> add single 1, then 0s
    padded = inputs + [1] + [0] * (state_size - len(inputs) - 1)
    # update state
    state = np.array(padded, dtype=np.int64)

    # apply functions through rounds
    for i in range(num_rounds):
        # s-box layer
        state = np.array([s_box_cpu(curr_state) for curr_state in state], dtype=np.int64)
        # MDS matricurr_state multiplication #1
        state = mds_multiply_cpu(state)
        # add round constant #1
        state = np.mod(state + round_constants_cpu[2 * i], p)
        # inverse s-box layer
        state = np.array([inv_s_box_cpu(curr_state) for curr_state in state], dtype=np.int64)
        # MDS matricurr_state multiplication #2
        state = mds_multiply_cpu(state)
        # Add round constant #2
        state = np.mod(state + round_constants_cpu[2 * i + 1], p)
    # return first rate elements of final state
    return list(state[:rate])

##### GPU Functions #####
### S-Bocurr_state Function
def s_box_gpu(curr_state):
    # curr_state^alpha mod p
    return pow(curr_state, alpha, p)
### Inverse S-Box Function
def inv_s_box_gpu(curr_state):
    # curr_state^alpha_inv mod p
    return pow(curr_state, inv_alpha, p)

def mds_multiply_gpu(state):
    # multiply state by MDS matricurr_state
    return cp.mod(mds_matrix_gpu @ state, p)

### Rescue Hash Functions
def rescue_hash_gpu(input_array: cp.ndarray) -> cp.ndarray:
    """
    Args:
        input_array: CuPy 1D array of integers (each a scalar input)

    Returns:
        CuPy array of shape (len(input_array), rate)
    """
    batch_size = input_array.shape[0]

    # === Pad to full state ===
    ones = cp.ones((batch_size, 1), dtype=cp.uint64)
    zeros = cp.zeros((batch_size, state_size - 2), dtype=cp.uint64)  # 1 input + 1 one + rest zeros
    print("Input array", input_array)
    state = cp.concatenate([input_array[:, None], ones, zeros], axis=1)

    # === Main Rescue Rounds ===
    for i in range(num_rounds):
        # S-box layer
        state = cp.power(state, alpha, dtype=cp.uint64) % p
        # MDS mix
        state = cp.mod(state @ mds_matrix_gpu.T, p)
        # Add round constant #1
        rc1 = round_constants_gpu[2 * i][None, :]  # shape (1, state_size)
        state = cp.mod(state + rc1, p)

        # Inverse S-box
        state = cp.power(state, inv_alpha, dtype=cp.uint64) % p
        state = cp.mod(state @ mds_matrix_gpu.T, p)

        # Add round constant #2
        rc2 = round_constants_gpu[2 * i + 1][None, :]
        state = cp.mod(state + rc2, p)
    print(state)
    # Return first `rate` values of each state
    return state[:, :1].reshape(-1)

### Test Cases
if __name__ == "__main__":

    # test input on larger scale
    # test_input = [16, 234]  # small test
    test_input = np.arange(1000, dtype=np.int64)
    test_input_gpu = cp.array(test_input, dtype=cp.uint64)

    ### CPU test
    # empty input
    empty_test = rescue_hash_cpu([])
    print("Rescue hash of []:", empty_test)

    # non-empty input
    non_empty_test = rescue_hash_cpu([16, 234])
    print("Rescue hash of [16, 234]:", non_empty_test)

    # ensure hash function is deterministic
    if ( rescue_hash_cpu([45, 125]) == rescue_hash_cpu([45, 125]) ):
        print("Deterministic")
    else:
        print("Not deterministic")

    ### GPU test
    # empty input
    empty_test = rescue_hash_gpu(cp.array([0], dtype=cp.uint64))
    print("Rescue hash of []:", empty_test)

    # non-empty input
    non_empty_test = rescue_hash_gpu(cp.array([16, 234], dtype=cp.uint64))
    print("Rescue hash of [16, 234]:", non_empty_test)

    # ensure hash function is deterministic
    if cp.all( rescue_hash_gpu(cp.array([45, 125], dtype=cp.uint64)) == rescue_hash_gpu(cp.array([45, 125], dtype=cp.uint64)) ):
        print("Deterministic")
    else:
        print("Not deterministic")

    ######## code for comparing execution times
    ######## CPU is faster than GPU, need to process hash in batches
    ### CPU test
    start_cpu = time.time()
    # cpu_result = rescue_hash_cpu(test_input)  # for small test
    cpu_result = []
    for x in test_input:
        hash_val = rescue_hash_cpu([x])[0]
        cpu_result.append(hash_val)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    
    ### GPU test
    start_gpu = time.time()
    gpu_result = rescue_hash_gpu(test_input_gpu)
    cp.cuda.Device(0).synchronize() # finish GPU completely
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu
    
    ### Output
    print("\nExecution Time Comparison")
    print(f"CPU Result: {cpu_result}")
    print(f"GPU Result: {gpu_result}")
    print(f"CPU Time: {cpu_time:.6f} seconds")
    print(f"GPU Time: {gpu_time:.6f} seconds")