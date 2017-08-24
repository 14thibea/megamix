cdef void dist_matrix_update(double [:,:] points, double [:,:] means,double [:,:] dist, double [:,:] dist_matrix)

cdef class Kmeans:
    #Attributes
    cdef str name
    cdef str init
    cdef int n_components
    cdef double kappa
    cdef int _is_initialized
    cdef int iteration
    cdef int window
    
    cdef double [:,:] N
    cdef double [:,:] X
    cdef double [:,:] log_weights
    cdef double [:,:] means

    # Temporary memoryviews
    cdef double [:,:] N_temp            # (n_components,)
    cdef double [:,:] X_temp_fortran    # (n_components,dim)
    cdef double [:,:] X_temp            # (n_components,dim)
    cdef double [:,:] dist_matrix       # (window,n_components)
    cdef double [:,:] dist              # (n_components,dim)
    
    # Methods
    cdef void _check_parameters(self)
    cdef void _step_E_gen(self, double [:,:] points, double [:,:] assignements,
                          double [:,:] dist_matrix)
    cpdef void _step_M(self,double[:,:] points,double[:,:] assignements)
        