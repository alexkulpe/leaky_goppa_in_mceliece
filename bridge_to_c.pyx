#distutils: language = c
#distutils: extra_compile_args = -march=native -O3
from libc.stdint cimport uint64_t

cdef extern from "utilsc.c":
    uint64_t next_subset(uint64_t x, uint64_t limit)

def sage_next_subset(x, limit):
    cdef uint64_t x_data = x
    cdef uint64_t limit_data = limit
    return next_subset(x_data, limit_data)