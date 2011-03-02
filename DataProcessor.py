import tables
import numpy
import os

def matfile_test():
    run = tables.file.openFile(os.path.join('matlab', '00170.h5'))
    return run
