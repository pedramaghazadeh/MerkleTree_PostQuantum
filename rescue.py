# Imports
import numpy as np
import cupy as cp
print(cp.__version__)
print(cp.cuda.runtime.getDeviceCount())

### Initializations
# prime for finite field (same as merkle tree) should be large, like 2**259 - 19
p = 2**255 - 19
# s-box ecurr_stateponent (adds nonlinearity)
alpha = 5
# inverse s-box ecurr_stateponent (mod p - 1)
inv_alpha = pow(alpha, -1, p - 1)
# number of rescue rounds
num_rounds = 10
# state size = input length + capacity (e.g. 2 + 1 = 3)
state_size = 3
# number of elements in output
rate = 2
# MDS matrix curr_state -> micurr_state state variables between rounds, introduces diffusion
mds_matrix = np.array([
    [2, 3, 1],
    [1, 1, 4],
    [3, 5, 6]
], dtype=object)
# round constants (generate unique values)
round_constants = [np.array([(i * j + 1) % p for j in range(state_size)], dtype=object) for i in range(2 * num_rounds)]

### S-Box curr_state Function
def sbocurr_state(curr_state):
    # curr_state^alpha mod p
    return pow(curr_state, alpha, p)
### Inverse S-Box Function
def inv_s_box(curr_state):
    # curr_state^alpha_inv mod p
    return pow(curr_state, inv_alpha, p)

def mds_multiply(state):
    # multiply state by MDS matricurr_state
    return np.mod(mds_matrix @ state, p)

### Rescue Hash Function
def rescue_hash(inputs):
    if type(inputs) is not list:
        inputs = [inputs] # ensure inputs is a list
    # pad inputs -> add single 1, then 0s
    padded = inputs + [1] + [0] * (state_size - len(inputs) - 1)
    # update state
    state = np.array(padded, dtype=object)

    # apply functions through rounds
    for i in range(num_rounds):
        # s-box layer
        state = np.array([sbocurr_state(curr_state) for curr_state in state], dtype=object)
        # MDS matricurr_state multiplication #1
        state = mds_multiply(state)
        # add round constant #1
        state = np.mod(state + round_constants[2 * i], p)
        # inverse s-box layer
        state = np.array([inv_s_box(curr_state) for curr_state in state], dtype=object)
        # MDS matricurr_state multiplication #2
        state = mds_multiply(state)
        # Add round constant #2
        state = np.mod(state + round_constants[2 * i + 1], p)
    # return first rate elements of final state
    return list(state[:rate])

### Test Cases
if __name__ == "__main__":

    # empty input
    empty_test = rescue_hash([])
    print("Rescue hash of []:", empty_test)

    # non-empty input
    non_empty_test = rescue_hash([16, 234])
    print("Rescue hash of [16, 234]:", non_empty_test)

    # ensure hash function is deterministic
    if ( rescue_hash([45, 125]) == rescue_hash([45, 125]) ):
        print("Deterministic")
    else:
        print("Not deterministic")