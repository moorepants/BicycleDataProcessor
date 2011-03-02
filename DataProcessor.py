import tables
import numpy
import os

def matfile_test():
    tables.file.fileopen(os.path.join('matlab', '00170.mat'))
    return tables.root.NIData.read()
