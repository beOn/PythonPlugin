#!/usr/bin/env python

# for testing stuff out before adding it to the plugin

import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt

f = 4.0
fs = 30000.0
ds1 = 3000.0
ds2 = 300.0
dsf1 = int(fs/ds1);
dsf2 = int(ds1/ds2);



a = np.array([np.sin(2.0*np.pi*f*t) for t in np.arange(0.0,10.0,1.0/fs)])
print(np.arange(0.0,10.0,1.0/fs)[-1])

# we'll decimate the signal using an IIR filter in two steps: first from 30k to
# 3k, then to 300
# TODO: get sampling rate from plugin object var
a = decimate(a, dsf1, zero_phase=True)
a = decimate(a, dsf2, zero_phase=True)



# plt.plot(a)
# plt.plot(a)
# plt.show()

f = np.abs(np.fft.rfft(a[0:300]))
# plt.plot(f)
# plt.show()

print(len(f))
print(len(np.fft.rfftfreq(300)))
