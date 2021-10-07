# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx.dot(w)
    
    grad = -1/len(tx) * np.transpose(tx).dot(e)
    
    return grad


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size = 10):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        
            stoch_gradient = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
        
            loss = compute_loss(minibatch_y,minibatch_tx,w)
        
            w = w - gamma * stoch_gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
            
    return losses, ws