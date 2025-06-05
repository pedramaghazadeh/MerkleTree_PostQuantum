# Imports
import numpy as np
import cupy as cp
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
mds_matrix = np.array([
    [2, 3, 1],
    [1, 1, 4],
    [3, 5, 6]
], dtype=object)

# Round constants
round_constants = [np.array([(i * j + 1) % p for j in range(state_size)], dtype=object) for i in range(2 * num_rounds)]

##### CPU Functions #####
### S-Box Function
def sbox_cpu(curr_state):
    # curr_state^alpha mod p
    return pow(int(curr_state), alpha, p)
# Inverse S-Box function
def inv_s_box_cpu(curr_state):
    # curr_state^alpha_inv mod p
    return pow(int(curr_state), inv_alpha, p)

# MDS matrix multiplication
def mds_multiply_cpu(state):
    # multiply state by MDS matricurr_state
    return np.mod(mds_matrix @ state, p)

# Rescue hash function
def rescue_hash_cpu(inputs):
    # if type(inputs) is not list:
    #     inputs = [inputs]   # ensure inputs is a list
    # pad inputs -> add single 1, then 0s
    padded = inputs + [1] + [0] * (state_size - len(inputs) - 1)
    # update state
    state = np.array(padded, dtype=object)

    # apply functions through rounds
    for i in range(num_rounds):
        # s-box layer
        state = np.array([sbox_cpu(curr_state) for curr_state in state], dtype=object)
        # MDS matrix multiplication #1
        state = mds_multiply_cpu(state)
        # add round constant #1
        state = np.mod(state + round_constants[2 * i], p)
        # inverse S-box layer
        state = np.array([inv_s_box_cpu(curr_state) for curr_state in state], dtype=object)
        # MDS matrix multiplication #2
        state = mds_multiply_cpu(state)
        # add round constant #2
        state = np.mod(state + round_constants[2 * i + 1], p)
    # return first rate elements of final state
    return list(state[:rate])

# Batch hash function for multiple inputs (comparing execution time)
def batch_rescue_hash_cpu(input_list):
    # perform hash function on each input
    return [rescue_hash_cpu(inputs) for inputs in input_list]

# Test cases
if __name__ == "__main__":

    # imports sanity check
    print(cp.__version__)
    print(cp.cuda.runtime.getDeviceCount())
    
    # empty input
    start_time = time.perf_counter()
    empty_test = rescue_hash_cpu([])
    end_time = time.perf_counter()
    empty_time_ms = (end_time - start_time) * 1000
    print("CPU rescue hash of []:", empty_test)
    print(f"Time: {empty_time_ms:.3f} ms")
    
    # non-empty input #1
    start_time = time.perf_counter()
    non_empty_test_1 = rescue_hash_cpu([16, 234])
    end_time = time.perf_counter()
    non_empty_1_time_ms = (end_time - start_time) * 1000
    print("CPU rescue hash of [16, 234]:", non_empty_test_1)
    print(f"Time: {non_empty_1_time_ms:.3f} ms")
    
    # non-empty input #2
    start_time = time.perf_counter()
    non_empty_test_2 = rescue_hash_cpu([100, 200])
    end_time = time.perf_counter()
    non_empty_2_time_ms = (end_time - start_time) * 1000
    print("CPU rescue hash of [100, 200]:", non_empty_test_2)
    print(f"Time: {non_empty_2_time_ms:.3f} ms")
    
    # batch of 10000 random inputs
    np.random.seed(10)  # for reproducibility
    batch_inputs = np.random.randint(0, p, size=(10000, 2)).tolist() # randomize
    start_time = time.perf_counter()
    batch_test = batch_rescue_hash_cpu(batch_inputs)
    end_time = time.perf_counter()
    batch_time_ms = (end_time - start_time) * 1000
    print("CPU batch hash (10000 random inputs):", batch_test[:3])  # display first 3 only
    print(f"Time: {batch_time_ms:.3f} ms")
    
    # deterministic sanity check
    is_deterministic = rescue_hash_cpu([45, 125]) == rescue_hash_cpu([45, 125])
    if is_deterministic:
        print("CPU hash is deterministic")
    else:
        print("CPU hash is NOT deterministic")