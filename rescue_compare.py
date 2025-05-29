# Imports
import numpy as np
import cupy as cp
import math
import time
print(cp.__version__)
print(cp.cuda.runtime.getDeviceCount())

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
], dtype=cp.int64)
# round constants (generate unique values)
    # both CPU and GPU versions
round_constants_cpu = [np.array([(i * j + 1) % p for j in range(state_size)], dtype=np.int64) for i in range(2 * num_rounds)]
round_constants_gpu = [cp.array([(i * j + 1) % p for j in range(state_size)], dtype=cp.int64) for i in range(2 * num_rounds)]

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
def rescue_hash_gpu(inputs):
    # pad inputs -> add single 1, then 0s
    padded = inputs + [1] + [0] * (state_size - len(inputs) - 1)
    # update state
    state = cp.array(padded, dtype=cp.int64)

    # apply functions through rounds
    for i in range(num_rounds):
        # s-box layer
        state = cp.array([s_box_gpu(curr_state) for curr_state in state], dtype=cp.int64)
        # MDS matricurr_state multiplication #1
        state = mds_multiply_gpu(state)
        # add round constant #1
        state = cp.mod(state + round_constants_gpu[2 * i], p)
        # inverse s-box layer
        state = cp.array([inv_s_box_gpu(curr_state) for curr_state in state], dtype=cp.int64)
        # MDS matricurr_state multiplication #2
        state = mds_multiply_gpu(state)
        # Add round constant #2
        state = cp.mod(state + round_constants_gpu[2 * i + 1], p)
    # return first rate elements of final state
    return list(state[:rate])

### Test Cases
if __name__ == "__main__":

    # test input on larger scale
    test_input = [16, 234]

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
    empty_test = rescue_hash_gpu([])
    print("Rescue hash of []:", empty_test)

    # non-empty input
    non_empty_test = rescue_hash_gpu([16, 234])
    print("Rescue hash of [16, 234]:", non_empty_test)

    # ensure hash function is deterministic
    if ( rescue_hash_gpu([45, 125]) == rescue_hash_gpu([45, 125]) ):
        print("Deterministic")
    else:
        print("Not deterministic")

    ######## code for comparing execution times
    ######## CPU is faster than GPU, need to process hash in batches
    ### CPU test
    start_cpu = time.time()
    cpu_result = rescue_hash_cpu(test_input)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    
    ### GPU test
    start_gpu = time.time()
    gpu_result = rescue_hash_gpu(test_input)
    cp.cuda.Device(0).synchronize() # finish GPU completely
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu
    
    ### Output
    print("\nExecution Time Comparison")
    print(f"CPU Result: {cpu_result}")
    print(f"GPU Result: {gpu_result}")
    print(f"CPU Time: {cpu_time:.6f} seconds")
    print(f"GPU Time: {gpu_time:.6f} seconds")