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
                print("I'm here!")
                self.Enabled = 1

                f = 4.0
                fs = 100.0
                a = np.array([np.sin(2.0*np.pi*f*t) for t in np.arange(0.0,10.0,1.0/fs)])

                plt.plot(a)
                plt.show(block=false)
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
                return []
        def bufferfunction(self, n_arr):
                """Access to voltage data buffer. Returns events"""


                events = []
                return events
        def handleEvents(eventType,sourceID,subProcessorIdx,timestamp,sourceIndex):
                """handle events passed from OE"""
        def handleSpike(self, electrode, sortedID, n_arr):
                """handle spikes passed from OE"""


pluginOp = blp_trace()

include '../plugin.pyx'

