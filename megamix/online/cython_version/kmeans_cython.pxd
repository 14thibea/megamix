cdef void dist_matrix_update(double [:,:] points, double [:,:] means,double [:,:] dist, double [:,:] dist_matrix)

cdef class Kmeans:
    #Attributes
    cdef char* name
    cdef int n_components
    cdef double kappa
    cdef int n_jobs
    cdef int _is_initialized
    cdef int iteration
    cdef int window
    
    cdef double [:,:] N
    cdef double [:,:] X
    cdef double [:,:] means

    # Temporary memoryviews
    cdef double [:,:] N_temp            # (n_components,)
    cdef double [:,:] X_temp_fortran    # (n_components,dim)
    cdef double [:,:] X_temp            # (n_components,dim)
    cdef double [:,:] dist_matrix       # (window,n_components)
    cdef double [:,:] dist              # (n_components,dim)
    
    # Methods
    cdef void _check_parameters(self)
#    cdef void _step_E(self, double [:,:] points, int dim,
#                      double [:,:] assignements, double [:,:] dist_matrix)
#    cdef void _step_M(self,double[:,:] points,int dim,double[:,:] assignements)
        