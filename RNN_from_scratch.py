import numpy as np

data_size, vocab_size = 100, 2

hidden_size = 3
seq_length = 100
learning_rate = 1e-2

W1 = np.random.randn(hidden_size, vocab_size) * 0.01 #input to hidden
Wr = np.random.randn(hidden_size, hidden_size) * 0.01 #input to hidden
W2 = np.random.randn(vocab_size, hidden_size) * 0.01 #input to hidden
b1 = np.zeros((hidden_size, 1))
b2 = np.zeros((vocab_size, 1))

def generate_data(N=100):
    n = N
    theta = np.linspace(-2*np.pi, 2*np.pi * np.pi, n).reshape(n, 1)
    x = np.concatenate((0.25 * np.cos(theta), 0.25 * np.sin(theta)), axis=1) 
    y = np.roll(x, -1, axis=0)
    return x, y

x,y = generate_data(N=100)


inputs, targets = {},{}
for i in range(len(x)):
    inputs[i] = x[i,:].reshape(-1,1)
    targets[i] = y[i,:].reshape(-1,1)
    

def lossFun(inputs, targets, hprev):
    u1, h, u2, y_hat = {}, {}, {}, {}

    h[-1] = np.copy(hprev)
    
    loss = 0
    
    # forward:                                                                                                                                                                              
    for t in range(len(inputs)):
        u1[t] = np.dot(W1, inputs[t]) + np.dot(Wr, h[t-1]) + b1
        h[t] = np.tanh(u1[t]) 
        u2[t] = np.dot(W2, h[t]) + b2
        y_hat[t] = np.tanh(u2[t])
        loss += (y_hat[t] - targets[t])**2
    
    # backward:
    dW1, dWr, dW2 = np.zeros_like(W1), np.zeros_like(Wr), np.zeros_like(W2)
    db1, db2 = np.zeros_like(b1), np.zeros_like(b2)
    dhnext = np.zeros_like(h[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(y_hat[t])
        dy[t] = y_hat[t] - targets[t]
        du = (1 - hs[t]**2)
        dW2 += np.dot(dy*du, h[t].T)
        db2 += dy*du
    
        dh = np.dot(W1.T, dy) + dhnext
        dhraw = (1 - h[t]**2) * dh
        db1 += dhraw
        dW1 += np.dot(dhraw, targets[t].T)
        dWr += np.dot(dhraw, h[t-1].T) 
        dhnext = np.dot(Wr.T, dhraw) 
    for dparam in [dW1, dWr, dW2, db1, db2]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients                                                                                                                 
    return loss, dW1, dWr, dW2, db1, db2, h[len(inputs)-1]

hprev = np.zeros((hidden_size,1)) # reset RNN memory
loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)