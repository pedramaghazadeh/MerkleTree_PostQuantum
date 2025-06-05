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
], dtype=cp.int64)

# Round constants (convert from CPU to GPU version)
round_constants = [np.array([(i * j + 1) % p for j in range(state_size)], dtype=object) for i in range(2 * num_rounds)]
round_constants = [cp.array(rc, dtype=cp.int64) for rc in round_constants]

##### GPU Functions #####
# S-Box function
def sbox(curr_state):
    # curr_state^alpha mod p
    return pow(int(curr_state), alpha, p)
# Inverse S-Box function
def inv_s_box(curr_state):
    # curr_state^alpha_inv mod p
    return pow(int(curr_state), inv_alpha, p)

# MDS matrix multiplication
def mds_multiply(state):
    # multiply state by MDS matricurr_state
    return cp.mod(cp.dot(mds_matrix, state), p)

# Rescue hash function (single input)
def rescue_hash(inputs):
    if not isinstance(inputs, list):
        inputs = [inputs]  # ensure inputs is a list
    # pad inputs -> add single 1, then 0s
    padded = inputs + [1] + [0] * (state_size - len(inputs) - 1)
    # update state
    state = cp.array(padded, dtype=cp.int64)

    # apply functions through rounds
    for i in range(num_rounds):
        # s-box layer
        state_np = cp.asnumpy(state)
        state = cp.array([sbox(curr_state) for curr_state in state_np], dtype=cp.int64)
        # MDS matrix multiplication #1
        state = mds_multiply(state)
        # add round constant #1
        state = cp.mod(state + round_constants[2 * i], p)
        # inverse S-box layer
        state_np = cp.asnumpy(state)
        state = cp.array([inv_s_box(curr_state) for curr_state in state_np], dtype=cp.int64)
        # MDS matrix multiplication #2
        state = mds_multiply(state)
        # add round constant #2
        state = cp.mod(state + round_constants[2 * i + 1], p)
    # return first rate elements of final state as Python list
    return cp.asnumpy(state[:rate]).tolist()

# Batch hash function for multiple inputs (comparing execution time)
    # parallel processing to take advantage of CUDA GPU
def batch_rescue_hash(input_list):
    # convert inputs to CuPy array with padding
    padded = np.array([inputs + [1] + [0] * (state_size - len(inputs) - 1) for inputs in input_list])
    states = cp.array(padded, dtype=cp.int64)  # (batch_size, state_size)

    # apply functions through rounds
    for i in range(num_rounds):
        # s-box layer
        states_np = cp.asnumpy(states)
        states_np = np.array([[sbox(curr_state) for curr_state in state] for state in states_np])
        states = cp.array(states_np, dtype=cp.int64)
        # MDS matrix multiplication #1
        states = cp.mod(cp.dot(mds_matrix, states.T).T, p)  # Adjusted for batched operation
        # add round constant #1
        states = cp.mod(states + round_constants[2 * i], p)
        # inverse S-box layer
        states_np = cp.asnumpy(states)   # CPU to GPU conversion
        states_np = np.array([[inv_s_box(curr_state) for curr_state in state] for state in states_np])
        states = cp.array(states_np, dtype=cp.int64)
        # MDS matrix multiplication #2
        states = cp.mod(cp.dot(mds_matrix, states.T).T, p)
        # add round constant #2
        states = cp.mod(states + round_constants[2 * i + 1], p)
    # return first rate elements of final states
    return cp.asnumpy(states[:, :rate]).tolist()

# Test cases
if __name__ == "__main__":

    # imports sanity check
    print(cp.__version__)
    print(cp.cuda.runtime.getDeviceCount())
    
    # empty input
    start_time = time.perf_counter()
    empty_test = rescue_hash([])
    end_time = time.perf_counter()
    empty_time_ms = (end_time - start_time) * 1000
    print("GPU rescue hash of []:", empty_test)
    print(f"Time: {empty_time_ms:.3f} ms")
    
    # non-empty input #1
    start_time = time.perf_counter()
    non_empty_test_1 = rescue_hash([16, 234])
    end_time = time.perf_counter()
    non_empty_1_time_ms = (end_time - start_time) * 1000
    print("GPU rescue hash of [16, 234]:", non_empty_test_1)
    print(f"Time: {non_empty_1_time_ms:.3f} ms")
    
    # non-empty input #2
    start_time = time.perf_counter()
    non_empty_test_2 = rescue_hash([100, 200])
    end_time = time.perf_counter()
    non_empty_2_time_ms = (end_time - start_time) * 1000
    print("GPU rescue hash of [100, 200]:", non_empty_test_2)
    print(f"Time: {non_empty_2_time_ms:.3f} ms")
    
    # batch of 10000 random inputs
    np.random.seed(10)  # for reproducibility
    batch_inputs = np.random.randint(0, p, size=(10000, 2)).tolist() # randomize
    start_time = time.perf_counter()
    batch_results = batch_rescue_hash(batch_inputs)
    end_time = time.perf_counter()
    batch_time_ms = (end_time - start_time) * 1000
    print("GPU batch hash (10000 random inputs):", batch_results[:3]) # display first 3 only
    print(f"Time: {batch_time_ms:.3f} ms")
    
    # deterministic sanity check
    is_deterministic = rescue_hash([45, 125]) == rescue_hash([45, 125])
    if is_deterministic:
        print("GPU hash is deterministic")
    else:
        print("GPU hash is NOT deterministic")