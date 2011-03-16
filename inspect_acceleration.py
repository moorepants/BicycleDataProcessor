import tables as tab
import numpy as np
import matplotlib.pyplot as plt
import DataProcessor as dp

datafile = tab.openFile('InstrumentedBicycleData.h5')
datatable = datafile.root.data.datatable
for x in datatable.iterrows():
    if x['RunID'] == 4:
        pass
    else:
        if x['Maneuver'] != 'System Test':
            numSamp = x['NINumSamples']    
            sampleRate = x['NISampleRate']
            time = np.linspace(0., numSamp/sampleRate, num=numSamp)
            acceleration = dp.unsize_vector(x['FrameAccelY'], numSamp)
            print '--------------------'
            print 'Run ID:', x['RunID']
            print 'Speed:', x['Speed']
            print 'Notes:', x['Notes']
            print 'Environment:', x['Environment'] 
            print 'Maneuver:', x['Maneuver'] 
            print 'Total time:', time[-1]
            print 'Time of max value:', time[np.argmax(acceleration)]
            print 'Max value:', np.max(acceleration) 
            print '--------------------'
            if time[np.argmax(acceleration)] > 5.:
                plt.figure(x['RunID'])
                plt.plot(time, acceleration)
                plt.title(x['Speed'])

plt.show()
datafile.close()
