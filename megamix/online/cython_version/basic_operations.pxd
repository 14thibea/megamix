# Updates for sufficient statistics
cdef void update1D(double [:] a, int a_len, double b, double [:] result) nogil
cdef void update2D(double [:,:] a, int a_x, int a_y, double gamma, double [:,:] result) nogil
cdef void update3D(double [:,:,:] a, int a_x, int a_y, int a_z, double gamma, double [:,:,:] result) nogil

# Simple functions for arrays
cdef double sum2D(double [:,:] a,int a_x, int a_y) nogil
cdef void add2Dand2D(double [:,:] a, int a_x, int a_y, double [:,:] b, int b_x, int b_y, double [:,:] result) nogil
cdef void add2Dscalar_reduce(double [:,:] a, int a_x, int a_y, double b, double[:,:] result) nogil
cdef void add2Dscalar(double [:,:] a, int a_x, int a_y, double b, double[:,:] result) nogil
cdef void add2Dscalar_col_i(double [:,:] a, int a_x, int a_y, double b, double[:,:] result) nogil
cdef void subtract2Dby2D(double [:,:] a, int a_x, int a_y, double [:,:] b, int b_x, int b_y, double [:,:] result) nogil
cdef void subtract2Dby2D_idx(double [:,:] a, int a_x, int a_y, double [:,:] b, int axis, int idx, double [:,:] result) nogil
cdef void multiply2Dby1D(double [:,:] a, int a_x, int a_y, double [:] b, double [:,:] result) nogil
cdef void multiply2Dbyvect2D(double [:,:] a, int a_x, int a_y, double [:,:] b, int axis, double [:,:] result) nogil
cdef void multiply2Dby2D_idx(double [:,:] a, int a_x, int a_y, double [:,:] b, int axis, int idx, double [:,:] result) nogil
cdef void multiply2Dbyscalar(double [:,:] a, int a_x, int a_y, double b, double [:,:] result) nogil
cdef void multiply3Dbyvect2D(double [:,:,:] a, int a_x, int a_y, int a_z, double [:,:] b, double [:,:,:] result)
cdef void multiply3Dbyscalar(double [:,:,:] a, int a_x, int a_y, int a_z, double b, double [:,:,:] result) nogil
cdef void divide2Dbyscalar(double [:,:] a, int a_x, int a_y, double b, double [:,:] result) nogil
cdef void divide2Dby1D(double [:,:] a, int a_x, int a_y, double [:] b, double [:,:] result) nogil
cdef void divide2Dbyvect2D(double [:,:] a, int a_x, int a_y, double [:,:] b, double [:,:] result) nogil
cdef void divide3Dbyscalar(double [:,:,:] a, int a_x, int a_y, int a_z, double b, double [:,:,:] result) nogil
cdef void divide3Dbyvect2D(double [:,:,:] a, int a_x, int a_y, int a_z, double [:,:] b, double [:,:,:] result) nogil
cdef void initialize(double [:,:] a, int a_x, int a_y) nogil
cdef void reciprocal2D(double [:,:] a, int a_x, int a_y, double [:,:] result) nogil
cdef void exp2D(double [:,:] a, int a_x, int a_y, double[:,:] result) nogil
cdef void log2D(double [:,:] a, int a_x, int a_y, double[:,:] result) nogil
cdef void sqrt2D(double [:,:] a, int a_x, int a_y, double[:,:] result) nogil

# Functions for the E step
cdef int argmin(double [:,:] a, int a_x, int a_y) nogil
cdef void norm_axis1(double [:,:] a, int a_x, int a_y, double [:,:] result) nogil
cdef void norm_axis1_matrix(double [:,:] a, int a_x, int a_y,
                            double [:,:] b, int b_x, int b_y,
                            double [:,:] diff, double [:,:] result) nogil  

# Transposition                    
cdef void transpose(double [:,:] a, int a_len, int a_wid, double [:,:] result) nogil

# Using fortran arrays
cdef void transpose_spe_f2c(double[:,:] a, int a_len, int a_wid, double [:,:] result) nogil
cdef void dot_spe_c(double [:,:] a, int a_len, int a_wid, double [:,:] b, int b_len, int b_wid, double [:,:] result) nogil
cdef void dot_spe_c2(double [:,:] a, int a_len, int a_wid, double [:,:] b, int b_len, int b_wid, double [:,:] result) nogil
cdef void true_slice(double [:,:] a, int a_x, int a_wid, double [:,:] b, int b_len) nogil
                    
                    
# Functions for covariances
cdef void reg_covar(double [:,:,:] cov, int idx, int dim, double reg_covar) nogil
cdef void transpose_spe_f2c_and_write(double[:,:] a, int a_len, int a_wid, double [:,:,:] result, int i) nogil
cdef void triangular_inverse_cov(double[:,:,:] a, int idx, int dim, double [:,:] result) nogil
cdef void cast2Din3D(double[:,:] a, int idx, int dim, double [:,:,:] result) nogil
cdef void cast3Din2D(double[:,:,:] a, int idx, int dim, double [:,:] result) nogil
cdef double log_det_tr(double [:,:] a, int dim) nogil
cdef void writecol_sum_square(double [:,:] a, int a_x, int a_y, int axis, int idx, double [:,:] result) nogil
cdef void erase_above_diag(double [:,:] a, int dim) nogil
                          
# Specific functions for VBGMM
cdef void special_psi_sum(double [:,:] nu, int n_components, int dim, double [:,:] result) nogil
