# This file includes SGD and Adam for parameter update.
import numpy as np


def sgd(w, dw, params={}):
    """
    Perform Vanilla SGD for parameter update.
    Arguments:
        w: numpy array of current weight
        dw: numpy array of gradient of loss w.r.t. current weight
        params: dictionary containing hyper-parameters
            - lr: float of learning rate
    Outputs:
        next_w: updated weight
        params: updated dictionary of hyper-parameters
    """
    # set default parameters
    params.setdefault('lr', 1e-2)
    # update w
    next_w = w - params['lr'] * dw

    return next_w, params


def adam(w, dw, params={}):
    """
    Perform Adam update rule for parameter update.
    This update rule incorporates moving averages of both the gradient and its square and a bias correction term.
    Arguments:
        w: numpy array of current weight
        dw: numpy array of gradient of loss w.r.t. current weight
        params: dictionary containing hyper-parameters
            - lr: float of learning rate
            - beta1: float of decay rate for moving average of first moment of gradient
            - beta2: float of decay rate for moving average of second moment of gradient
            - epsilon: float of a small value used for smoothing to avoid dividing by zero
            - m: numpy array of moving average of gradient with the sameshape of w
            - v: moving average of squared gradient with the sameshape of w
            - t: int of iteration number
    Outputs:
        next_w: updated weight
        params: updated dictionary of hyper-parameters
    """
    # set default parameters
    params.setdefault('lr', 1e-2)
    params.setdefault('beta1', 0.9)
    params.setdefault('beta2', 0.999)
    params.setdefault('epsilon', 1e-8)
    params.setdefault('m', np.zeros_like(w))
    params.setdefault('v', np.zeros_like(w))
    params.setdefault('t', 0)
    # update w
    lr, beta1, beta2, epsilon, m, v, t = \
        params['lr'], params['beta1'], params['beta2'], params['epsilon'], params['m'], params['v'], params['t']
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * dw ** 2
    t += 1
    alpha = params['lr'] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    w -= alpha * (m / (np.sqrt(v) + epsilon))
    params['t'] = t
    params['m'] = m
    params['v'] = v
    next_w = w

    return next_w, params
