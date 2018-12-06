import numpy as np
cimport numpy as np

EMSGS = {1:'n must be >= 1',
         2:'x must be sorted in ascending order'}

# defined in _dip.c
cdef extern nogil:

    void diptst(const double* x,
                const int* n_,
                double* dip,
                int* lo_hi,
                int* ifault,
                int* gcm,
                int* lcm,
                int* mn,
                int* mj,
                const int* min_is_0,
                const int* debug)

# wrapper to expose 'diptst' C function to Python space
def _dip(x, full_output, min_is_0, x_is_sorted, debug):

    cdef:
        double[:] x_
        int n = x.shape[0]
        double dip = np.nan
        int[:] lo_hi = np.empty(4, dtype=np.int32)
        int ifault = 0
        int[:] gcm = np.empty(n, dtype=np.int32)
        int[:] lcm = np.empty(n, dtype=np.int32)
        int[:] mn = np.empty(n, dtype=np.int32)
        int[:] mj = np.empty(n, dtype=np.int32)
        int min_is_0_ = min_is_0
        int debug_ = debug

    # input needs to be sorted in ascending order
    if not x_is_sorted:

        # force a copy to prevent inplace modification of input
        x = x.copy()

        # sort inplace
        x.sort()

    # cast to double
    x_ = x.astype(np.double)

    diptst(&x_[0], &n, &dip, &lo_hi[0], &ifault, &gcm[0], &lcm[0], &mn[0],
           &mj[0], &min_is_0_, &debug_)

    if ifault:
        raise ValueError(EMSGS[ifault])

    if full_output:
        res_dict = {
            'xs':np.array(x_),
            'n':n,
            'dip':dip,
            'lo':lo_hi[0],
            'hi':lo_hi[1],
            'xl':x_[lo_hi[0]],
            'xu':x_[lo_hi[1]],
            'gcm':np.array(gcm[:lo_hi[2]]),
            'lcm':np.array(lcm[:lo_hi[3]]),
            'mn':np.array(mn),
            'mj':np.array(mj),
        }
        return dip, res_dict

    else:
        return dip

