# -*- coding: utf-8 -*-
# @Author: Ben Acland
# @Date:   2019-04-02 0:38:00

import sys
from os.path import dirname, join, abspath
import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt

from cython cimport view

sys.path.append('/Users/benacland/devdir/PythonPlugin/python_modules')
sys.path.append('/Users/benacland/devdir/PythonPlugin/python_modules/blp_trace') # TODO put the python path in the C++ executalbe

# from .multiproc_base import (BaseMultiprocPlugin, BasePlotController,
#     ConstantLengthCircularBuff, DownsamplingThread)

include './multiproc_base.pyx' # TODO: put this in a proper module

isDebug = False

# =================================
# =           Constants           =
# =================================

ds1 = 3000
ds2 = 300
FFT_CHUNK_SIZE = ds2
FFT_NFREQS = int(ds2//2 + 1)
FREQS = np.fft.rfftfreq(FFT_CHUNK_SIZE, 1/float(ds2))
PLOT_SECS = 300 # we'll show 5 min of data for starters

# ==============================
# =           Plugin           =
# ==============================
        
class BLPSpectPlotPlugin(BaseMultiprocPlugin):
    def __init__(self):
        super(BLPSpectPlotPlugin, self).__init__()

    # override init_controller
    def init_controller(self, input_frequency):
        """
        Subclasses should override this method to set self.controller to an
        instance of a concrete subclass of BasePlotController.
        """
        self.controller = BLPSpectPlotController(input_frequency, plot_frequency=0.1)
    
    # TODO: override plugin_name
    def plugin_name(self):
        """Subclasses should override to tell us their name
        
        Returns:
            string: The name of this plugin
        """
        return "BLP Spect Plot"

# =============================================
# =           Subprocess Controller           =
# =============================================

class BLPSpectPlotController(BasePlotController):
    def __init__(self, *args, **kwargs):
        super(BLPSpectPlotController, self).__init__(*args, **kwargs)
        self.Enabled = 1    # TODO: needed?
        self.plt_chan = 1
        # plotting objects
        self.ax = None
        self.figure = None
        self.mesh = None
        self.plt_timer = None
        # set up the buffer for power estimates (300 secs @ 1 Hz = 300)
        self.est_buff = ConstantLengthCircularBuff(np.float64, int((ds2/FFT_CHUNK_SIZE)*PLOT_SECS))
    
    # -----------  Overrides  -----------
    
    def init_preprocessors(self):
        # TODO: delete this print statement
        print("input freq is {}".format(self.input_frequency))
        # set up decimating preprocessors to go from 30 kHz to 300 Hz in two steps
        ds1 = DownsamplingThread(self.input_frequency, 3000, self.pipe_reader.buffer)
        ds2 = DownsamplingThread(ds1.fsOut, 300, ds1.output_buff)
        self.preprocessors = [ds1, ds2]

    def start_plotting(self):
        # set up the plot
        self.figure, self.ax = plt.subplots()
        self.mesh = self.ax.pcolormesh(
            np.array(range(PLOT_SECS+1)),
            FREQS[1:],
            np.zeros((len(FREQS)-1,PLOT_SECS)))
        self.ax.margins(y=0.1)
        self.ax.set_xlim(0., PLOT_SECS)
        self.ax.set_ylim(FREQS[1], FREQS[-1])

        # start the callback timer
        self.plt_timer = self.figure.canvas.new_timer(interval=500,)
        self.plt_timer.add_callback(self.gui_callback)
        self.plt_timer.start()
        plt.show()

    def stop_plotting(self):
        # stop the timer and close the plot
        self.plt_timer.stop()
        self.plt_timer = None
        plt.close('all')

    def update_plot(self):
        # if channel has changed, clear the buffers
        if self.params.get('chan_in', self.plt_chan) != self.plt_chan:
            self.plt_chan = self.params['chan_in']
            self.est_buff.fill(0.0)

        # estimate power for data from input buffer (if there's enough)
        nChunks = int(np.floor(self.plot_input_buffer.nUnread / FFT_CHUNK_SIZE))
        # not enough? then there's nothing to do.
        if nChunks == 0:
            return
        for i in range(nChunks):
            self.est_buff.write(np.abs(np.fft.rfft(self.plot_input_buffer.read(FFT_CHUNK_SIZE)[self.plt_chan,:])).reshape(FFT_NFREQS,1))

        # set the data like so:
        C = self.est_buff.read().ravel()
        self.mesh.set_array(C)

        # update the color bar limits like so:
        self.mesh.set_clim(vmin=np.min(C), vmax=np.max(C))

        # redraw the plot
        self.ax.margins(y=0.1) # TODO: remove if you can
        self.ax.relim()
        self.ax.autoscale_view(True,True,True)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # -----------  Static Overrides  -----------

    @staticmethod
    def param_config():
        # subclasses may override if they want to surface UI elements in the OE GUI
        chan_labels = list(range(32))
        return (("int_set", "chan_in", chan_labels),)

pluginOp = BLPSpectPlotPlugin()
include '../plugin.pyx'
