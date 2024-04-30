import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def genRandMat(row, column, rand_range):
    return np.random.uniform(-rand_range, rand_range, (row, column))

def genWeights(in_s: int, hidden_s: list, out_s:int):
    w = [genRandMat(hidden_s[0], in_s, 1)]
    for _ in range(len(hidden_s) - 1):
        w.append(genRandMat(hidden_s[_ + 1], hidden_s[_], 1))
    w.append(genRandMat(out_s, hidden_s[-1], 1))
    return w

def forwardProp(input_mat, weights):
    activations = [input_mat]
    # Forward matrix multiplication -> get activation of each layers
    for weight in weights:
        input_mat = sigmoid(weight @ input_mat)
        activations.append(input_mat)
    return activations

def backProp(activation, weights, l_rate, layer_ind, diff):
    # Backprop finished
    if layer_ind == 0:
        return True
    
    # Take infomation for backprop
    this_l_a = activation[layer_ind]
    this_l_w = weights[layer_ind - 1]
    last_l_a = activation[layer_ind - 1]
    # This layer's weights adjustment
    this_l_w -= l_rate * (diff * (this_l_a*(1 - this_l_a))) @ np.transpose(last_l_a)     
    # Backprop activation changes from this layer -> last layer
    del_a = np.transpose(l_rate * np.transpose(diff * (this_l_a*(1 - this_l_a))) @ this_l_w)

    return backProp(activation, weights, l_rate, layer_ind - 1, del_a)


rate = 0.1

inp = 4
inp_mat = genRandMat(inp, 1, 1)

out = 4

# Expected output
c = np.random.randint(0,2,(out , 1))

h_l = [3,3,3]

w = genWeights(inp, h_l, out)

for i in range(10000):
    a = forwardProp(inp_mat, w)
    backProp(a, w, rate, len(a) - 1, a[-1] - c)
    print(np.sum((forwardProp(inp_mat, w)[-1] - c)**2))

print(w)
