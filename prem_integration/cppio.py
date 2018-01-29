import struct

import numpy as np


DTYPES = {
    'uint8': 'B',
    'int32': 'i',
    'float32': 'f',
    'float64': 'd',
}


def write(filename, arr, dtype=None):
    """
    Writes the array data as a binary file. The header contains 3 int32 values
    describing the shape of the array followed by a char representing the data
    type (either 'i', 'f' or 'd' for 'int32', 'float32' and 'float64'
    respectively). The rest is data in the described format.

    """
    if dtype is not None:
        arr = arr.astype(dtype)
    # extract the dimensions and make sure there are 3 of them (d, h, w)
    dims = arr.shape
    if len(dims) == 2:
        arr = arr.reshape([arr.shape[0], arr.shape[1], 1])
        dims = arr.shape
    f = open(filename, 'wb')
    # write dimensions
    for dim in arr.shape:
        f.write(struct.pack('I', dim))
    # write datatype
    dtype_char = DTYPES[str(arr.dtype)]
    f.write(struct.pack('c', dtype_char))

    fmt = dtype_char * dims[2]
    # write data
    for row in range(dims[0]):
        for col in range(dims[1]):
            f.write(struct.pack(fmt, *arr[row, col, :]))
    f.close()
