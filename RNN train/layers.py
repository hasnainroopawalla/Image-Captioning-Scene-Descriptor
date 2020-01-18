# This file defines different layers used for RNN and for image captioning.
import numpy as np


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    There is no shape requirement for input x.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single time stamp in Vanilla RNN with a tanh activation function:
    next_h = tanh(Wx * x + Wh * prev_h + b).
    Arguments:
        x: input data for current time stamp with shape (N, D)
        prev_h: hidden state from previous time stamp with shape (N, H)
        Wx: weight matrix for input data with shape (D, H)
        Wh: weight matrix for hidden states with shape (H, H)
        b: bias with shape (H,)
    Outputs:
        next_h: hidden state after the forward step with shape (N, H)
        cache: cache used for back-prop
    """
    next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)
    cache = x, prev_h, Wx, Wh, b, next_h
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Run the backward pass for a single time stamp in Vanilla RNN with a tanh activation function:
    dx = (1 - next_h^2) * Wx * dnext_h
    dprev_h = (1 - next_h^2) * Wh * dnext_h
    dWx = (1 - next_h^2) * x.T * dnext_h
    dWh = (1 - next_h^2) * h.T * dnext_h
    db = (1 - next_h^2) * dnext_h
    Arguments:
        dnext_h: gradient of hidden state with shape (N, H)
        cache: cache used for back-prop
    Outputs:
        dx: gradient of input data with shape (N, D)
        dprev_h: gradient of hidden state for previous time stamp with shape (N, H)
        dWx: gradient of weight matrix for input data with shape (D, H)
        dWh: gradient of weight matrix for hidden states with shape (H, H)
        db: gradient of bias with shape (H,)
    """
    x, prev_h, Wx, Wh, b, h = cache
    dtanh = (1 - h ** 2) * dnext_h
    dx = np.dot(dtanh, Wx.T)
    dprev_h = np.dot(dtanh, Wh.T)
    dWx = np.dot(x.T, dtanh)
    dWh = np.dot(h.T, dtanh)
    db = np.sum(dtanh, axis=0)
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a forward pass for vanilla RNN on an entire sequence of data.
    The input has N sequences, each of which is composed of T vectors, each of dimension D.
    The hidden state size for the RNN is H.
    Arguments:
        x: input data for with shape (N, T, D)
        h0: initial hidden state with shape (N, H)
        Wx: weight matrix for input data with shape (D, H)
        Wh: weight matrix for hidden states with shape (H, H)
        b: bias with shape (H,)
    Outputs:
        h: hidden states after the forward step with shape (N, T, H)
        cache: cache used for back-prop
    """
    N, T, D = x.shape
    _, H = h0.shape
    x = np.swapaxes(x, 0, 1)  # swap axes for easier loops
    h = np.zeros((T, N, H))  # initialize h
    prev_h = h0
    cache = []
    for t in range(T):
        next_h, cache_ = rnn_step_forward(x[t], prev_h, Wx, Wh, b)
        prev_h = next_h
        cache.append(cache_)
        h[t] = prev_h
    h = np.swapaxes(h, 0, 1)  # swap axes for correct format
    return h, cache


def rnn_backward(dh, cache):
    """
    Run a backward pass for vanilla RNN from the gradient of all hidden states dh.
    Arguments:
        dh: gradient of all hidden states with shape (N, T, H)
        cache: cache used for back-prop
    Outputs:
        dx: gradient of input data with shape (N, T, D)
        dh0: gradient of initial hidden state with shape (N, H)
        dWx: gradient of weight matrix for input data with shape (D, H)
        dWh: gradient of weight matrix for hidden states with shape (H, H)
        db: gradient of bias with shape (H,)
    """
    dh = dh.copy()  # this is very important!
    N, T, H = dh.shape
    D = cache[0][0].shape[-1]  # extract parameter D fro initialization
    dh = np.swapaxes(dh, 0, 1)  # swap axes for easier loops
    # initializations for derivatives
    dx, dh0, dWx, dWh, db = np.zeros((T, N, D)), np.zeros((N, H)), np.zeros((D, H)), np.zeros((H, H)), np.zeros((H,))
    for t in range(T):
        dx[t], dprev_h, dWx_, dWh_, db_ = rnn_step_backward(dh[t], cache[t])
        # update parameters
        dh[t] += dprev_h
        dWx += dWx_
        dWh += dWh_
        db += db_
    dh0 = dprev_h
    dx = np.swapaxes(dx, 0, 1)  # swap axes for correct format
    return dx, dh0, dWx, dWh, db


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Run the forward pass for a single time stamp in LSTM:
    a = Wx * x + Wh * h + b;
    a = [a_i, a_f, a_o, a_g];
    i, f, o, g = sigmoid(a_i), sigmoid(a_f), sigmoid(a_o), tanh(a_g);
    next_c = f ⊙ prev_c + i ⊙ g;
    next_h = o ⊙ tanh(next_c).
    The shapes are consistent with Vallina RNN.
    Arguments:
        x: input data for current time stamp with shape (N, D)
        prev_h: hidden state from previous time stamp with shape (N, H)
        prev_c: cell state from previous time stamp with shape (N, H)
        Wx: weight matrix for input data with shape (D, 4H)
        Wh: weight matrix for hidden states with shape (H, 4H)
        b: bias with shape (4H,)
    Outputs:
        next_h: hidden state after the forward step with shape (N, H)
        next_c: cell state after the forward step with shape (N, H)
        cache: cache used for back-prop

    """
    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    H = a.shape[1] // 4
    a_i, a_f, a_o, a_g = a[:, :H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:]
    i, f, o, g = sigmoid(a_i), sigmoid(a_f), sigmoid(a_o), np.tanh(a_g)
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)
    cache = x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_h, next_c
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Run the backward pass for a single time stamp in LSTM:
    do = dnext_h ⊙ tanh(next_c);
    ddnext_c = dnext_h ⊙ o;
    df = dnext_c ⊙ prev_c;
    dprev_c = dnext_c ⊙ f;
    di = dnext_c ⊙ g;
    dg = dnext_c ⊙ i;
    da_i = di ⊙ i ⊙ (1 - i);
    da_f = df ⊙ f ⊙ (1 - f);
    da_o = do ⊙ o ⊙ (1 - o);
    da_g = dg ⊙ (1 - g^2);
    da = [da_i, da_f, da_o, da_g];
    dx = da * Wx;
    dh = da * Wh;
    dWx = da * x.T;
    dWh = da * h.T;
    db = da.
    Arguments:
        dnext_h: gradient of hidden state with shape (N, H)
        dnext_c: gradient of cell state with shape (N, H)
        cache: cache used for back-prop
    Outputs:
        dx: gradient of input data with shape (N, D)
        dprev_h: gradient of hidden state with shape (N, H)
        dprev_c: gradient of cell state with shape (N, H)
        dWx: gradient of weight matrix for input data with shape (D, 4H)
        dWh: gradient of weight matrix for hidden states with shape (H, 4H)
        db: gradient of bias with shape (4H,)
    """
    x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, h, c = cache
    do = dnext_h * np.tanh(c)
    dnext_c += dnext_h * o * (1 - np.tanh(c) ** 2)
    dprev_c = dnext_c * f
    dg = dnext_c * i
    di = dnext_c * g
    df = dnext_c * prev_c
    da_i = di * i * (1 - i)
    da_f = df * f * (1 - f)
    da_o = do * o * (1 - o)
    da_g = dg * (1 - g ** 2)
    da = np.concatenate((da_i, da_f, da_o, da_g), axis=-1)
    dx = np.dot(da, Wx.T)
    dWx = np.dot(x.T, da)
    dprev_h = np.dot(da, Wh.T)
    dWh = np.dot(prev_h.T, da)
    db = np.sum(da, axis=0)
    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Run a forward pass for LSTM on an entire sequence of data.
    The dimensions are consistent with Vallina RNN.
    Arguments:
        x: input data for with shape (N, T, D)
        h0: initial hidden state with shape (N, H)
        Wx: weight matrix for input data with shape (D, H)
        Wh: weight matrix for hidden states with shape (H, H)
        b: bias with shape (H,)
    Outputs:
        h: hidden states after the forward step with shape (N, T, H)
        cache: cache used for back-prop
    """
    N, T, D = x.shape
    _, H = h0.shape
    x = np.swapaxes(x, 0, 1)  # swap axes for easier loops
    h = np.zeros((T, N, H))
    prev_c = np.zeros((N, H))
    prev_h = h0
    cache = []
    for i in range(T):
        prev_h, prev_c, cache_ = lstm_step_forward(x[i], prev_h, prev_c, Wx, Wh, b)
        h[i] = prev_h
        cache.append(cache_)
    h = np.swapaxes(h, 0, 1)  # swap back for correct format
    return h, cache


def lstm_backward(dh, cache):
    """
    Run a backward pass for LSTM from the derivative of all hidden states dh.
    Arguments:
        dh: gradient of all hidden states with shape (N, T, H)
        cache: cache used for back-prop
    Outputs:
        dx: gradient of input data with shape (N, T, D)
        dh0: gradient of initial hidden state with shape (N, H)
        dWx: gradient of weight matrix for input data with shape (D, H)
        dWh: gradient of weight matrix for hidden states with shape (H, H)
        db: gradient of bias with shape (H,)
    """
    dh = dh.copy()  # very important!
    N, T, H = dh.shape
    D = cache[0][0].shape[-1]  # extract parameter D
    dh = np.swapaxes(dh, 0, 1)  # swap axes for easier loops
    # initialization of derivatives
    dx, dWx, dWh, db, dprev_h = np.zeros((T, N, D)), np.zeros((D, 4*H)), np.zeros((H, 4*H)), np.zeros((4*H,)), np.zeros((N, H))
    dprev_c = np.zeros(dprev_h.shape)
    for t in reversed(range(T)):
        dh[t] += dprev_h
        dx[t], dprev_h, dprev_c, dWx_, dWh_, db_ = lstm_step_backward(dh[t], dprev_c, cache[t])
        dWx += dWx_
        dWh += dWh_
        db += db_
    dh0 = dprev_h
    dx = np.swapaxes(dx, 0, 1)
    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, W, b):
    """
    Run a forward pass for temporal affine layer. The dimensions are consistent with RNN/LSTM forward passes.
    Arguments:
        x: input data with shape (N, T, D)
        W: weight matrix for input data with shape (D, M)
        b: bias with shape (M,)
    Outputs:
        out: output data with shape (N, T, M)
        cache: cache for back-prop
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = np.dot(x.reshape(N * T, D), W).reshape(N, T, M) + b
    cache = x, W, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Run a backward pass for temporal affine layer. The dimensions are consistent with RNN/LSTM forward passes.
    Arguments:
        dout: gradient of output data with shape (N, T, M)
        cache: cache for back-prop
    Outputs:
        dx: gradient of input data with shape (N, T, D)
        dW: gradient of weight matrix with shape (D, M)
        db: gradient of bias with shape (M,)
    """
    x, W, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]
    dx = np.dot(dout.reshape(N * T, M), W.T).reshape(N, T, D)
    dw = np.dot(dout.reshape(N * T, M).T, x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))
    return dx, dw, db


def temporal_softmax_loss(x, y, mask):
    """
    This function is adapted from CS231n.
    A temporal version of softmax loss for use in RNNs.
    The vocabulary has size V for each time step of a time series of length T, with a batch size of N.
    Cross-entropy loss is calculated, summed and averaged over all time steps across the batch.
    Arguments:
    - x: input scores for all vocabulary elements with shape of (N, T, V)
    - y: ground-truth indices at each time step with shape of (N, T), each element of which is in [0, V)
    - mask: boolean array with shape of (N, T) indicating whether the scores at x[n, t] should contribute to the loss
    Outputs:
    - loss: float of loss
    - dx: gradient of loss with respect to scores x
    """
    N, T, V = x.shape
    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]
    dx = dx_flat.reshape(N, T, V)
    return loss, dx


def word_embedding_forward(x, W):
    """
    Run a forward pass for word embeddings.
    The dimensions are consistent with parameters in temporal softmax loss.
    Arguments:
    - x: integer array with shape of (N, T) giving indices of words, each of which lies in [0, V)
    - W: weight matrix with shape of (V, D) giving word vectors for all words.
    Outputs:
    - out: array with shape of (N, T, D) giving word vectors for all input words.
    - cache: cache for back-prop
    """
    out = W[x, :]
    cache = x, W
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Run a backward pass for word embeddings.
    The dimensions are consistent with parameters in temporal softmax loss.
    Arguments:
    - dout: gradient of output with shape of (N, T, D)
    - cache: cache used for back-prop
    Outputs:
    - dW: gradient of weight matrix with shape of (V, D)
    """
    x, W = cache
    dW = np.zeros(W.shape)
    np.add.at(dW, x, dout)
    return dW
