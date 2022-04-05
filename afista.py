import numpy as np
import pylops

def soft_thresholding(x, threshold):
    return np.maximum( np.abs( x ) - threshold, 0. ) * np.sign( x )

def AFISTA(Op, b, niter, alpha, eps, x_true, decay=None, acceleration=True, weight_update=1, delta=1e-10):
    # initialize variables
    m, n = Op.shape
    y = np.zeros(n)
    weights = np.ones(n)
    x_thresh_old = np.zeros(n)
    x_thresh_new = np.zeros(n)
    threshold = alpha * eps * 0.5
    t_old = 1
    cost = np.zeros(niter)
    res_norm = np.zeros(niter)
    # prevent division by zero
    do_update = 1
    if decay is None:
        decay = np.ones(niter)
    
    # start AFISTA
    for i in range(niter):

        # residual
        res = ( Op * y - b )
        res_norm[i] = np.linalg.norm(res)
        
        # gradient step
        x = y - alpha * Op.H * res
        
        # thresholding
        x_thresh_new = soft_thresholding( x, threshold * weights )
                           
        # update the weights for AFISTA. w_k = 1/|x_k|
        if acceleration and do_update == weight_update:
            weights = 1 / ( np.abs( x_thresh_new ) + delta )
            do_update = 0
        
        # Nesterov step
        t_new = 1/2 + 1/2 * np.sqrt( 1 + 4 * t_old ** 2 )
        y = x_thresh_new + ( t_old - 1 ) / t_new * ( x_thresh_new - x_thresh_old )
        
        # update the previous solution
        x_thresh_old = x_thresh_new
        t_old = t_new
        
        # calculate the error
        cost[i] = np.linalg.norm( y - x_true ) / np.linalg.norm( x_true )
        
        # count to see when to do the update
        do_update += 1
        
    return y, cost, res_norm


