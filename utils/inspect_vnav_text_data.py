import tables as tab
import numpy as np
import DataProcessor as dp

datafile = tab.openFile('InstrumentedBicycleData.h5')
datatable = datafile.root.data.datatable

nanList = []

for x in datatable.iterrows():
    cell = x['AccelerationX']
    vnSampRate = x['NINumSamples']
    vnSig = dp.unsize_vector(cell, vnSampRate)
    numNan = np.sum(np.isnan(vnSig))
    if numNan > 2:
        nanList.append((x['RunID'], numNan))

nanList.sort(key=lambda x: x[1])
for thing in nanList:
    print thing
