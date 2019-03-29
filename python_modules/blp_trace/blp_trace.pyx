import sys
import numpy as np
cimport numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from cython cimport view

isDebug = False

plt.ion()

class blp_trace(object):
        def __init__(self):
                """initialize object data"""
                self.Enabled = 1
                self.channel = 1

                f = 4.0
                fs = 100.0
                a = np.array([np.sin(2.0*np.pi*f*t) for t in np.arange(0.0,10.0,1.0/fs)])

                plt.plot(a)
                plt.show()
        def startup(self, sr):
                """to be run upon startup"""
        def plugin_name(self):
                """tells OE the name of the program"""
                return "blp_trace"
        def is_ready(self):
                """tells OE everything ran smoothly"""
                return self.Enabled
        def param_config(self):
                """return button, sliders, etc to be present in the editor OE side"""
                chan_labels = list(range(1,33));
                return [("int_set", "channel", chan_labels),]
        def bufferfunction(self, n_arr):
                """Access to voltage data buffer. Returns events"""
                cdef int chan_in
                cdef int chan_out
                chan_in = self.channel-1 # -1 because the list is 1-based
                cdef int n_samples = n_arr.shape[1]

                events = []
                return events
        def handleEvents(eventType,sourceID,subProcessorIdx,timestamp,sourceIndex):
                """handle events passed from OE"""
        def handleSpike(self, electrode, sortedID, n_arr):
                """handle spikes passed from OE"""


pluginOp = blp_trace()

include '../plugin.pyx'

