cpdef void cholupdate(double [:,:,:] cov_chol, int idx, double [:,:] points, double[:,:] points_temp)
cdef void logsumexp_axis(double[:,:] a, int a_x, int a_y, int axis, double [:,:] result)
cdef void _log_normal_matrix(double [:,:] points,double [:,:] means,
                             double [:,:,:] cov_chol,double [:,:] log_normal_matrix,
                             double [:,:] cov_temp,double [:,:] points_temp_fortran,
                             double [:,:] points_temp,double [:,:] mean_temp)
cdef class BaseMixture:
    cdef str name
    cdef str init
    cdef int _is_initialized
    cdef int n_components
    cdef int iteration
    cdef int window
    cdef double reg_covar
    cdef double kappa
    cdef list convergence_criterion_test

    # Only for VBGMM
    cdef double alpha_0
    cdef double beta_0
    cdef double nu_0
    cdef int alpha_0_flag
    cdef int beta_0_flag
    cdef int nu_0_flag
    cdef int mean_prior_flag
    cdef int inv_prec_prior_flag
    cdef double [:,:] inv_prec_prior    # (dim,dim)
    cdef double [:,:] mean_prior        # (1,dim)
    cdef double [:,:] alpha             # (1,n_components)
    cdef double [:,:] beta              # (1,n_components)
    cdef double [:,:] nu                # (1,n_components)
    
    # Parameters of all models
    cdef double [:,:] log_weights
    cdef double [:,:] means
    cdef double [:,:,:] cov
    cdef double [:,:,:] cov_chol
    
    # Sufficient statistics
    cdef double [:,:] N                     # (1,n_components)
    cdef double [:,:] X                     # (n_components,dim)
    cdef double [:,:,:] S                   # (n_components,dim,dim)
    
    #Temporary arrays
    cdef double [:,:] N_temp                # (1,n_components)
    cdef double [:,:] N_temp2               # (1,n_components)
    cdef double [:,:] X_temp_fortran        # (n_components,dim)
    cdef double [:,:] X_temp                # (n_components,dim)
    cdef double [:,:,:] S_temp              # (n_components,dim,dim)
    cdef double [:,:] cov_temp              # (dim,dim)
    cdef double [:,:] points_temp           # (window,dim)
    cdef double [:,:] points_temp2          # (window,dim)
    cdef double [:,:] resp_temp             # (window,n_components)
    cdef double [:,:] mean_temp             # (1,dim)
    cdef double [:,:] log_prob_norm         # (window,1) --> may be used as convergence criterion
    
    # Abstract methods
    cdef void _step_E_gen(self, double [:,:] points, double [:,:] log_resp,
                double [:,:] points_temp_fortran, double [:,:] points_temp,
                double [:,:] log_prob_norm)
    cdef void _cstep_E(self,double [:,:] points,double [:,:] log_resp)
    cpdef void _step_M(self)
    cpdef void _sufficient_statistics(self,double [:,:] points,double [:,:] log_resp)
    cdef double _convergence_criterion(self,double [:,:] points,double [:,:] log_resp,
                                       double [:,:] log_prob_norm)
    
    cdef void _check_common_parameters(self)
    cdef void _check_prior_parameters(self, points)
    cdef void _cinitialize_cov(self,double [:,:] points, double [:,:] assignements,
                              double [:,:] diff, double [:,:] diff_weighted)
    cdef void _cinitialize_weights(self,double [:,:] points,double [:,:] log_normal_matrix,
                            double [:,:] points_temp, double [:,:] points_temp_fortran)
    cdef void _compute_cholesky_matrices(self)
    cdef void _initialize_temporary_arrays(self,double [:,:] points)
    