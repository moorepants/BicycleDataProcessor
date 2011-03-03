import tables
import os

def importmat(filename):
    run = tables.file.openFile(os.path.join('matlab', filename))
    return run

