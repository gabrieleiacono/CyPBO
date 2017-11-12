from __future__ import division
cimport numpy as np
import numpy as np
from scipy.special import binom
from scipy.stats import gaussian_kde
from itertools import combinations
from libc.math cimport sqrt, log
import array
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef double mean(double[:] a) nogil:
    cdef:
        int dim = a.shape[0]
        double tot = 0.0
        int i

    for i in xrange(dim):
        tot += a[i]
    return tot/dim

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef double std(double[:] a) nogil:
    cdef:
        double v = 0.0
        double m
        int dim = a.shape[0]
        int i

    m = mean(a)
    for i in xrange(dim):
        v += (a[i] - m)**2
    
    return sqrt(v/dim)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef double sharpe(double[:] a) nogil:
    return mean(a)/std(a)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef double sortino_m30(np.ndarray[np.float64_t,ndim=1] a):
    return mean(a)/std(a[a<0])*sqrt(12096)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef double[:] compute_R(double[:,:] J):
    cdef:
        int N = J.shape[1]
        int i
        double[:] R = np.empty(N)
    
    for i in xrange(N):
        R[i] = sortino_m30(np.array(J[:,i]))
    
    return R

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cpdef double pbo(double[:,:] M, int S):
    """
    Computes the Probability of Backtest Overfitting and the Probaility of Loss.
    """

    
    cdef:
        int T, N, residual, sub_T, i, start, end

    T = M.shape[0]
    N = M.shape[1]
    residual = T % S
    if residual != 0:
        M = M[residual:,:]
        T = M.shape[0]
        N = M.shape[1]
    
    sub_T = T // S

    cdef:
        double[:,:,:] Ms = np.empty((S,sub_T,M.shape[1]))
        long[:] index = np.arange(M.shape[0],dtype=np.int64)
        long[:,:] index_s = np.empty((S,sub_T), dtype=np.int64)


    for i in xrange(S):
        start = i * sub_T
        end = (i + 1) * sub_T
        Ms[i,:,:] = M[start:end,:]
        index_s[i,:] = index[start:end]

    cdef:
        int combs = int(binom(S,S//2))
        int comb_index = 0
        int n_star
        double omega_bar
        double[:,:] J = np.empty((T/2, N))
        double[:,:] J_bar = np.empty((T/2,N))
        long[:] index_j =  np.empty(T/2, dtype=np.int64)
        long[:] index_bar = np.empty(T/2, dtype=np.int64)
        long[:] r_c = np.empty(N,dtype=np.int64)
        long[:] r_c_bar = r_c.copy()
        np.ndarray[np.float64_t, ndim=2] R_c = np.empty((combs,N))
        np.ndarray[np.float64_t, ndim=2] R_c_bar = R_c.copy()
        np.ndarray[np.float64_t, ndim=1] lambda_c = np.empty(int(binom(S,S//2)))
        double[:] R = np.empty(N)
        double[:] R_bar = R.copy()

    for c, d in zip(combinations(Ms, S//2), combinations(index_s, S//2)):
        
        # itertools.combinations is already preserving the order of c
        J = np.concatenate(c)
        index_j = np.concatenate(d)
        index_bar = np.setdiff1d(index,index_j)
        J_bar = np.take(M,index_bar, axis=0)
        R_c[comb_index,:] = compute_R(J)
        R_c_bar[comb_index,:] = compute_R(J_bar)
        r_c = np.array(R_c[comb_index,:]).argsort().argsort() + 1
        r_c_bar = np.array(R_c_bar[comb_index,:]).argsort().argsort() + 1
        
        n_star = np.array(r_c).argmax()
        omega_bar = r_c_bar[n_star] / (N + 1.0)
        lambda_c[comb_index] = log(omega_bar / (1 - omega_bar))
        comb_index += 1


    lambda_c = lambda_c[np.isfinite(lambda_c) & ~np.isnan(lambda_c)]
    pdf = gaussian_kde(lambda_c).pdf(np.arange(-10,0, 0.01))
    
    phi = np.trapz(pdf,dx=0.01)

    R = np.concatenate(R_c)
    R_bar = np.concatenate(R_c_bar)


    pdf_R_bar = gaussian_kde(R_bar).pdf(np.arange(-10,0,0.01))

    prob_loss = np.trapz(pdf_R_bar,dx=0.01)
    print "pbo: ", phi
    print "probability of loss: ", prob_loss
    return phi
