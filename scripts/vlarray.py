import tables as tab
import numpy as np

run1 = np.ones(10)
run2 = np.ones(15)

f = tab.openFile('vlarray.h5', mode='w')

#vlarray = f.createVLArray(f.root, 'vlarray', tab.Int32Atom(), 'ragged array')

class RunTable(tab.IsDescription):
    num = tab.Int32Col()
    data = tab.VLArray

table = f.createTable(f.root, 'table', RunTable)

row = table.row

for i, data in enumerate([run1, run2]):
    row['num'] = i
    row['data'].append(data)
    row.append()

table.flush()

