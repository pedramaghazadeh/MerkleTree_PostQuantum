# Imports
import cupy as cp
import numpy as np
import time

# Prime for finite field (ideally larger, but limited by CuPy)
p = 2**31 - 1

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

# MDS matrix curr_state -> micurr_state state variables between rounds, introduces diffusion
mds_matrix = cp.array([
    [2, 3, 1],
    [1, 1, 4],
    [3, 5, 6]
], dtype=cp.uint64)

# Round constants (convert from CPU to GPU version)
round_constants = [np.array([(i * j + 1) % p for j in range(state_size)], dtype=object) for i in range(2 * num_rounds)]
round_constants = [cp.array(rc, dtype=cp.uint64) for rc in round_constants]

##### GPU Functions #####
# S-Box function
def sbox(curr_state):
    # curr_state^alpha mod p
    return pow(int(curr_state), alpha, p)
# Inverse S-Box function
def inv_s_box(curr_state):
    # curr_state^alpha_inv mod p
    return pow(int(curr_state), inv_alpha, p)

# Vectorized Modular Exponentiation
def vectorized_mod_pow_gpu(base_array, exponent, modulus):
    # check if using CuPy arrays
    if isinstance(base_array, np.ndarray):
        base_array = cp.array(base_array, dtype=cp.uint64)

    # perform modular exponentiation
    result = cp.ones_like(base_array, dtype=cp.uint64)
    base = base_array % modulus
    exp = exponent
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % modulus
        base = (base * base) % modulus
        exp //= 2

    # return
    return result

# Vectorize the s-box and inverse s-box functions for batch processing with np
sbox_vectorized = np.vectorize(sbox)
inv_s_box_vectorized = np.vectorize(inv_s_box)

# MDS matrix multiplication
def mds_multiply(state):
    # multiply state by MDS matricurr_state
    return cp.mod(cp.dot(mds_matrix, state), p)

# Rescue hash function (single input)
def rescue_hash_gpu(inputs):
    inputs = inputs.get()
    # print(inputs)
    if not isinstance(inputs, list):
        inputs = list(inputs) # ensure inputs is a list
    
    # pad inputs -> add single 1, then 0s
    padded = inputs + [1] + [0] * (state_size - len(inputs) - 1)
    # update state
    state = cp.array(padded, dtype=cp.uint64)

    # apply functions through rounds
    for i in range(num_rounds):
        # s-box layer
        state_np = cp.asnumpy(state)
        state = cp.array([sbox(curr_state) for curr_state in state_np], dtype=cp.uint64)
        # MDS matrix multiplication #1
        state = mds_multiply(state)
        # add round constant #1
        state = cp.mod(state + round_constants[2 * i], p)
        # inverse S-box layer
        state_np = cp.asnumpy(state)
        state = cp.array([inv_s_box(curr_state) for curr_state in state_np], dtype=cp.uint64)
        # MDS matrix multiplication #2
        state = mds_multiply(state)
        # add round constant #2
        state = cp.mod(state + round_constants[2 * i + 1], p)
    # return first rate elements of final state as Python list
    return cp.array(state[:rate], dtype=cp.uint64)

# Batch hash function for multiple inputs (comparing execution time)
    # parallel processing to take advantage of CUDA GPU
def batch_rescue_hash_gpu(input_list):
    # print("Input list", input_list)
    # print("Batch input list:", input_list.shape)

    input_list = input_list.get()
    # print("Input list shape:", input_list.shape)
    if len(input_list.shape) == 1:
        input_list = np.reshape(input_list, (-1, 1)).tolist()  # ensure input_list is 2D
    # print("Reshaped batch input list:", input_list.shape)
    # convert inputs to CuPy array with padding
    padded = np.array([inputs + [1] + [0] * (state_size - len(inputs) - 1) for inputs in input_list])
    states = cp.array(padded, dtype=cp.uint64)  # (batch_size, state_size)

    # apply functions through rounds
    for i in range(num_rounds):
        # s-box layer
        ### old version
        #states_np = cp.asnumpy(states)
        #states_np = sbox_vectorized(states_np)  # Apply s-box to each element
        #states = cp.array(states_np, dtype=cp.uint64)
        ### NEW VECTORIZED MODULAR EXPONENTIATION
        states = vectorized_mod_pow_gpu(states, alpha, p)
        # MDS matrix multiplication #1
        states = cp.mod(cp.dot(mds_matrix, states.T).T, p)  # Adjusted for batched operation
        # add round constant #1
        states = cp.mod(states + round_constants[2 * i], p)
        # inverse S-box layer
        ### old version
        #states_np = cp.asnumpy(states)   # CPU to GPU conversion
        #states_np = inv_s_box_vectorized(states_np)  # Apply inverse s-box to each element
        #states = cp.array(states_np, dtype=cp.uint64)
        ### NEW VECTORIZED MODULAR EXPONENTIATION
        states = vectorized_mod_pow_gpu(states, inv_alpha, p)
        # MDS matrix multiplication #2
        states = cp.mod(cp.dot(mds_matrix, states.T).T, p)
        # add round constant #2
        states = cp.mod(states + round_constants[2 * i + 1], p)
    # return first rate elements of final states
    return cp.array(states[:, :rate], dtype=cp.uint64)  # (batch_size, rate)

# Test cases
if __name__ == "__main__":

    # imports sanity check
    print(cp.__version__)
    print(cp.cuda.runtime.getDeviceCount())
    
    # empty input
    start_time = time.perf_counter()
    empty_test = rescue_hash_gpu([])
    end_time = time.perf_counter()
    empty_time_ms = (end_time - start_time) * 1000
    print("GPU rescue hash of []:", empty_test)
    print(f"Time: {empty_time_ms:.3f} ms")
    
    # non-empty input #1
    start_time = time.perf_counter()
    non_empty_test_1 = rescue_hash_gpu([16, 234])
    end_time = time.perf_counter()
    non_empty_1_time_ms = (end_time - start_time) * 1000
    print("GPU rescue hash of [16, 234]:", non_empty_test_1)
    print(f"Time: {non_empty_1_time_ms:.3f} ms")
    
    # non-empty input #2
    start_time = time.perf_counter()
    non_empty_test_2 = rescue_hash_gpu([100, 200])
    end_time = time.perf_counter()
    non_empty_2_time_ms = (end_time - start_time) * 1000
    print("GPU rescue hash of [100, 200]:", non_empty_test_2)
    print(f"Time: {non_empty_2_time_ms:.3f} ms")
    
    # batch of 10000 random inputs
    np.random.seed(10)  # for reproducibility
    batch_inputs = np.random.randint(0, p, size=(10000, 1)).tolist() # randomize
    print(batch_inputs[:3])
    start_time = time.perf_counter()
    batch_results = batch_rescue_hash_gpu(batch_inputs)
    end_time = time.perf_counter()
    batch_time_ms = (end_time - start_time) * 1000
    print("GPU batch hash (10000 random inputs):", batch_results[:3]) # display first 3 only
    print(f"Time: {batch_time_ms:.3f} ms")
    
    # deterministic sanity check
    is_deterministic = rescue_hash_gpu([45, 125]) == rescue_hash_gpu([45, 125])
    if is_deterministic:
        print("GPU hash is deterministic")
    else:
        print("GPU hash is NOT deterministic")