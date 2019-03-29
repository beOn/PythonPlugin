#!/usr/bin/env python

# for testing stuff out before adding it to the plugin

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

f = 4.0
fs = 100.0
a = np.array([np.sin(2.0*np.pi*f*t) for t in np.arange(0.0,10.0,1.0/fs)])

plt.plot(a)
plt.show()

