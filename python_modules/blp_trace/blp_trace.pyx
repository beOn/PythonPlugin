import sys
import numpy as np
cimport numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt
from numpy_ringbuffer import RingBuffer

from cython cimport view

isDebug = False

# put the plot in interactive mode
plt.ion()

# constants
RAW_BUFF_LEN = 30000 * 100  # 100s buffer at 30kHz
DEC_BUFF_LEN = 300 * 110    # keep this one longer than the one above (in sec)
FFT_BUFF_LEN = 60
fs = 30000.0
ds1 = 3000.0
ds2 = 300.0
dsf1 = int(fs/ds1)
dsf2 = int(ds1/ds2)
DEC_CHUNK_SIZE = fs
FFT_CHUNK_SIZE = ds1
FFT_NFREQS = int(ds2//2 + 1)
FREQS = np.fft.rfftfreq(FFT_NFREQS)

class blp_trace(object):
    def __init__(self):
        """initialize object data"""
        self.Enabled = 1
        self.channel = 1

        # circular buffers
        self.rb = RingBuffer(capacity=RAW_BUFF_LEN, dtype=np.float64)
        self.db = RingBuffer(capacity=DEC_BUFF_LEN, dtype=np.float64)
        # TODO: this should be np.arrays of whatever size the fft gives you
        self.fb = RingBuffer(capacity=FFT_BUFF_LEN, dtype=(np.complex64,FFT_NFREQS))

        f = 4.0
        fs = 100.0
        a = np.array([np.sin(2.0*np.pi*f*t) for t in np.arange(0.0,10.0,1.0/fs)])

        plt.plot(a)
            plt.show()
    
    def startup(self, sr):
        """to be run upon startup"""
        # TODO: set, check the sampling rate
    
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
    
    def decimate(self):
        # decimate as much as you can from the raw buffer
        # goddammit, why isn't there a multi-popping circular buffer in numpy???
        while(len(self.rb) >= DEC_CHUNK_SIZE):
            # if you run out of space, clear the buffer by estimating
            if len(self.db) == DEC_BUFF_LEN:
                self.estimate()
            # decimate as much as you can without overflowing the next buffer
            avail = np.minimum(int(np.floor(DEC_BUFF_LEN-len(self.db), np.floor(len(self.rb)/float(DEC_CHUNK_SIZE)))))
            for i in range(avail):
                self.db.extend(decimate(decimate([self.rb.popleft() for x in range(DEC_CHUNK_SIZE)], dsf1, zero_phase=True), dsf2, zero_phase=True))
    
    def estimate(self):
        # estimate the fft for as many decimated chunks as you can
        while(len(self.db) >= FFT_CHUNK_SIZE):
            self.rb.append(np.abs(np.fft.rfft([self.db.popleft() for x in range(FFT_CHUNK_SIZE)])))

    
    def update_plot(self):
        # TODO: update the plot
        pass
    
    def bufferfunction(self, n_arr):
        """Access to voltage data buffer. Returns events"""
        cdef int chan_in
        cdef int chan_out
        chan_in = self.channel-1 # -1 because the list is 1-based
        cdef int n_samps = n_arr.shape[1]
        cdef int didx = 0

        if len(self.rb) < n_samps:
            # add everything to the buffer
            self.rb.extend(n_arr[chan_in, :])
        else:
            # decimate to clear out the buffer
            self.decimate()
            # repeatedly fill up what you can, then decimate, unitl you've got
            # everything in the buffer
            r = n_samps
            while r > 0:
                # TODO: this
                avail = np.minimum(r, RAW_BUFF_LEN-len(self.rb))
                self.rb.extend(narr[chan_in, n_samps-r:n_samps-r+avail])
                r = r-avail
                self.decimate()


        # if there's enough left to decimate, decimate what remains
        while(len(self.rb) > DEC_CHUNK_SIZE):
            # NOTE: protected by relative buffer size... but might need to add
            # similar logic to the above down the line
            self.decimate()


        # if there's enough to fft, fft and add to the fft buffer
        replot = len(self.db) > FFT_CHUNK_SIZE
        while(len(self.db) > FFT_CHUNK_SIZE):
            self.estimate()
            replot=True


        # if you fft-ed, update the figure
        if replot:
            self.update_plot()

        events = []
        return events
    def handleEvents(eventType,sourceID,subProcessorIdx,timestamp,sourceIndex):
        """handle events passed from OE"""
    def handleSpike(self, electrode, sortedID, n_arr):
        """handle spikes passed from OE"""


pluginOp = blp_trace()

include '../plugin.pyx'





# self.y = np.append(self.y, n_arr[self.chan_in-1, :])
