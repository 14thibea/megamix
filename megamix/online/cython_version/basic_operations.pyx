# cython: profile=True
from cython.parallel import parallel,prange
import cython
from scipy.special.cython_special cimport psi
from scipy.linalg.cython_blas cimport dgemm
from scipy.linalg.cython_lapack cimport dtrtri
from libc.math cimport log,exp,sqrt

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void transpose(double [:,:] a, int a_len, int a_wid, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_len):
        for y in xrange(a_wid):
            result[y,x] = a[x,y]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void transpose_spe_f2c(double[:,:] a, int a_len, int a_wid, double [:,:] result) nogil:
    cdef int x,y
    cdef int cptx = 0
    cdef int cpty = 0
    for x in xrange(a_len):
        for y in xrange(a_wid):
            if cptx >= a_len:
                cptx = 0
                cpty = cpty + 1
            result[cptx,cpty] = a[x,y]
            cptx = cptx + 1


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void update1D(double [:] a, int a_x, double gamma, double [:] result) nogil:
    cdef int x
    for x in xrange(a_x):
        result[x] = (1 - gamma) * result[x] + gamma * a[x]
        
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void update2D(double [:,:] a, int a_x, int a_y, double gamma, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            result[x,y] = (1 - gamma) * result[x,y] + gamma * a[x,y]
        
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void update3D(double [:,:,:] a, int a_x, int a_y, int a_z, double gamma, double [:,:,:] result) nogil:
    cdef int x,y,z
    for x in xrange(a_x):
        for y in xrange(a_y):
            for z in xrange(a_z):
                result[x,y,z] = (1 - gamma) * result[x,y,z] + gamma * a[x,y,z]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void divide2Dbyscalar(double [:,:] a, int a_x, int a_y, double b, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            result[x,y] = a[x,y] / b

               
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void divide2Dby1D(double [:,:] a, int a_x, int a_y, double [:] b, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            result[x,y] = a[x,y] / b[x]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void divide2Dbyvect2D(double [:,:] a, int a_x, int a_y, double [:,:] b, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            result[x,y] = a[x,y] / b[0,x]
            
  
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void divide3Dbyscalar(double [:,:,:] a, int a_x, int a_y, int a_z, double b, double [:,:,:] result) nogil:
    cdef int x,y,z
    for x in xrange(a_x):
        for y in xrange(a_y):
            for z in xrange(a_z):
                result[x,y,z] = a[x,y,z] / b
          
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void divide3Dbyvect2D(double [:,:,:] a, int a_x, int a_y, int a_z, double [:,:] b, double [:,:,:] result) nogil:
    cdef int x,y,z
    for x in xrange(a_x):
        for y in xrange(a_y):
            for z in xrange(a_z):
                result[x,y,z] = a[x,y,z] / b[0,x]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void multiply2Dby1D(double [:,:] a, int a_x, int a_y, double [:] b, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            result[x,y] = a[x,y] * b[x]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void multiply2Dbyvect2D(double [:,:] a, int a_x, int a_y, double [:,:] b, int axis, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            if axis==0:
                result[x,y] = a[x,y] * b[0,x]
            if axis==1:
                result[x,y] = a[x,y] * b[y,0]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void multiply2Dby2D_idx(double [:,:] a, int a_x, int a_y, double [:,:] b, int axis, int idx, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            if axis==0:
                result[x,y] = a[x,y] * b[idx,y]
            if axis==1:
                result[x,y] = a[x,y] * b[x,idx]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void multiply2Dbyscalar(double [:,:] a, int a_x, int a_y, double b, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            result[x,y] = a[x,y] * b
                  

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void multiply3Dbyvect2D(double [:,:,:] a, int a_x, int a_y, int a_z, double [:,:] b, double [:,:,:] result):
    cdef int x,y,z
    for x in xrange(a_x):
        for y in xrange(a_y):
            for z in xrange(a_z):
                result[x,y,z] = a[x,y,z]*b[0,x]
    

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void multiply3Dbyscalar(double [:,:,:] a, int a_x, int a_y, int a_z, double b, double [:,:,:] result) nogil:
    cdef int x,y,z
    for x in xrange(a_x):
        for y in xrange(a_y):
            for z in xrange(a_z):
                result[x,y,z] = a[x,y,z] * b

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void soustract2Dby2D(double [:,:] a, int a_x, int a_y,
                          double [:,:] b, int b_x, int b_y,
                          double [:,:] result) nogil:
    cdef int a_x_flag = 0
    cdef int a_y_flag = 0
    cdef int b_x_flag = 0
    cdef int b_y_flag = 0
    cdef int xmax = a_x
    cdef int ymax = a_y
    
    if a_x == 1:
        xmax = b_x
        a_x_flag = 1
    elif b_x == 1:
        b_x_flag = 1
        
    if a_y == 1:
        ymax = b_y
        a_y_flag = 1
    elif b_y == 1:
        b_y_flag = 1
    
    cdef int x,y
    cdef double value_a
    cdef double value_b
    for x in xrange(xmax):
        for y in xrange(ymax):
            if a_x_flag:
                value_a = a[0,y]
            elif a_y_flag:
                value_a = a[x,0]
            else:
                value_a = a[x,y]

            if b_x_flag:
                value_b = b[0,y]
            elif b_y_flag:
                value_b = b[x,0]
            else:
                value_b = b[x,y]
                
            result[x,y] = value_a - value_b


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void soustract2Dby2D_idx(double [:,:] a, int a_x, int a_y, double [:,:] b, int axis, int idx, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            if axis==0:
                result[x,y] = a[x,y] - b[idx,y]
            if axis==1:
                result[x,y] = a[x,y] - b[x,idx]
    
    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void reciprocal2D(double [:,:] a, int a_x, int a_y, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            result[x,y] = 1./a[x,y]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void initialize(double [:,:] a, int a_x, int a_y) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            a[x,y] = 0.
             
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int argmin(double [:,:] a, int a_x, int a_y) nogil:
    cdef int y = 0
    cdef double minimum = a[a_x,0]
    cdef int result = 0
    
    for y in xrange(a_y):
        if a[a_x,y] < minimum:
            result = y
            minimum = a[a_x,y]
    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double sum2D(double [:,:] a,int a_x, int a_y) nogil:
    cdef int x,y
    cdef double result = 0
    
    for x in xrange(a_x):
        for y in xrange(a_y):
            result += a[x,y]
            
    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void add2Dand2D(double [:,:] a, int a_x, int a_y,
                          double [:,:] b, int b_x, int b_y,
                          double [:,:] result) nogil:
    cdef int a_x_flag = 0
    cdef int a_y_flag = 0
    cdef int b_x_flag = 0
    cdef int b_y_flag = 0
    cdef int xmax = a_x
    cdef int ymax = a_y
    
    if a_x == 1:
        xmax = b_x
        a_x_flag = 1
    elif b_x == 1:
        b_x_flag = 1
        
    if a_y == 1:
        ymax = b_y
        a_y_flag = 1
    elif b_y == 1:
        b_y_flag = 1
    
    cdef int x,y
    cdef double value_a
    cdef double value_b
    for x in xrange(xmax):
        for y in xrange(ymax):
            if a_x_flag:
                value_a = a[0,y]
            elif a_y_flag:
                value_a = a[x,0]
            else:
                value_a = a[x,y]

            if b_x_flag:
                value_b = b[0,y]
            elif b_y_flag:
                value_b = b[x,0]
            else:
                value_b = b[x,y]
                
            result[x,y] = value_a + value_b

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)  # turn off negative index wrapping for entire function
cdef void add2Dscalar_reduce(double [:,:] a, int a_x, int a_y,
                             double b, double[:,:] result) nogil:
    cdef int x,y
    
    for y in xrange(a_y):
        result[0,y] = b
        for x in xrange(a_x):
            result[0,y] += a[x,y]
                  

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void add2Dscalar(double [:,:] a, int a_x, int a_y, double b, double[:,:] result) nogil:
    cdef int x,y
    
    for y in xrange(a_y):
        for x in xrange(a_x):
            result[x,y] = a[x,y]+b
                  
                  
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void add2Dscalar_col_i(double [:,:] a, int a_x, int a_y, double b, double[:,:] result) nogil:
    cdef int x,y
    
    for x in xrange(a_x):
        result[x,a_y] = a[x,a_y]+b


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void dot_spe_c2(double [:,:] a, int a_len, int a_wid, double [:,:] b, int b_len, int b_wid, double [:,:] result) nogil:
    '''
    The second component is transposed according to C the first is not
    '''
    cdef int m, n, k, lda, ldb, ldc
    cdef double alpha, beta
    
    m = a_len
    n = b_len
    k = a_wid
    alpha = 1.0
    beta = 0.0
    lda = k
    ldb = k
    ldc = m
    
    dgemm("T", "N", &m, &n, &k, &alpha, &a[0,0], &lda, &b[0,0], &ldb, &beta, &result[0,0], &ldc)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void dot_spe_c(double [:,:] a, int a_len, int a_wid, double [:,:] b, int b_len, int b_wid, double [:,:] result) nogil:
    '''
    The first component is transposed according to C the second is not
    '''
    cdef int m, n, k, lda, ldb, ldc
    cdef double alpha, beta
    
    m = a_wid
    n = b_wid
    k = a_len
    alpha = 1.0
    beta = 0.0
    lda = m
    ldb = n
    ldc = m
    
    dgemm("N", "T", &m, &n, &k, &alpha, &a[0,0], &lda, &b[0,0], &ldb, &beta, &result[0,0], &ldc)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void norm_axis1(double [:,:] a, int a_x, int a_y, double [:,:] result) nogil:
    cdef int i,j
    cdef double value
    
    for i in xrange(a_x):
        value = 0
        for j in xrange(a_y):
            value = value + a[i,j]**2
        result[0,i] = value**0.5
              
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void norm_axis1_matrix(double [:,:] a, int a_x, int a_y,
                            double [:,:] b, int b_x, int b_y,
                            double [:,:] diff, double [:,:] result) nogil:
    cdef int i
    
    for i in xrange(a_x):
        soustract2Dby2D(a[i:i+1:],1,a_y,b,b_x,b_y,diff)
        norm_axis1(diff,b_x,b_y,result[i:i+1:])


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void true_slice(double [:,:] a, int a_x, int a_wid, double [:,:] b, int b_len) nogil:
    cdef int y,x
    
    for x in xrange(b_len):
        for y in xrange(a_wid):
            b[x,y] = a[x+a_x,y]
            
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void reg_covar(double [:,:,:] cov, int idx, int dim, double reg_covar) nogil:
    
    for y in xrange(dim):
        cov[idx,y,y] += reg_covar
               
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void transpose_spe_f2c_and_write(double[:,:] a, int a_len, int a_wid, double [:,:,:] result, int i) nogil:
    cdef int x,y
    cdef int cptx = 0
    cdef int cpty = 0
    for x in xrange(a_len):
        for y in xrange(a_wid):
            if cptx >= a_len:
                cptx = 0
                cpty = cpty + 1
            result[i,cptx,cpty] = a[x,y]
            cptx = cptx + 1
        
        
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void triangular_inverse_cov(double[:,:,:] a, int idx, int dim, double [:,:] result) nogil:
    """
    This is working for lower triangular matrix in C
    """
    cdef int N = dim
    cdef int lda = dim
    cdef int info = 0
    cast3Din2D(a,idx,dim,result)
    
    dtrtri('U','N',&N,&result[0,0],&lda,&info)
    

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void cast3Din2D(double[:,:,:] a, int idx, int dim, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(dim):
        for y in xrange(dim):
            result[x,y] = a[idx,x,y]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void cast2Din3D(double[:,:] a, int idx, int dim, double [:,:,:] result) nogil:
    cdef int x,y
    for x in xrange(dim):
        for y in xrange(dim):
            result[idx,x,y] = a[x,y]
            

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double log_det_tr(double [:,:] a, int dim) nogil:
    cdef int x
    cdef double result = 0
    for x in xrange(dim):
        result += log(a[x,x])
        
    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void writecol_sum_square(double [:,:] a, int a_x, int a_y, int axis, int idx, double [:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            if axis == 0:
                result[y,idx] += a[x,y]**2
            elif axis==1:
                result[x,idx] += a[x,y]**2 
                      

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void erase_above_diag(double [:,:] a, int dim) nogil:
    cdef int x,y
    for x in xrange(dim):
        for y in xrange(x+1,dim):
            a[x,y] = 0
             

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void exp2D(double [:,:] a, int a_x, int a_y, double[:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            result[x,y] = exp(a[x,y])

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void log2D(double [:,:] a, int a_x, int a_y, double[:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            result[x,y] = log(a[x,y])
            
            
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void sqrt2D(double [:,:] a, int a_x, int a_y, double[:,:] result) nogil:
    cdef int x,y
    for x in xrange(a_x):
        for y in xrange(a_y):
            result[x,y] = sqrt(a[x,y])


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void special_psi_sum(double [:,:] nu, int n_components, int dim, double [:,:] result) nogil:
    cdef int x,y
    cdef double value
    for y in xrange(n_components):
        value = 0
        for x in xrange(dim):
            value += psi(0.5 * (nu[0,y] - x))
        result[0,y] = value
    
