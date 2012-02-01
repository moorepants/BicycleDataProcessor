from bicycledataprocessor import database
import os

def test_create_signal_table():
    db = database.DataSet(fileName='sigtest.h5')
    # this should not work because the file isn't open
    db.create_signal_table()
    # open a new file and create the table
    db.open(mode='w')
    db.create_signal_table()
    db.close()
    # now try to create the table if it is already there
    db.open(mode='a')
    db.create_signal_table()
    db.close()
    os.remove('sigtest.h5')
