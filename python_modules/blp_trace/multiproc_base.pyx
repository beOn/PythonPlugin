# -*- coding: utf-8 -*-
# @Author: Ben Acland
# @Date:   2019-04-01 16:22:51

import sys
import os
from os.path import dirname, join, abspath, isfile
from collections import deque
import numpy as np
cimport numpy as np
from cython cimport view
from threading import Thread, RLock, Event
from warnings import warn
from inspect import getargspec
from time import sleep
import multiprocessing as mp

# NOTE: move if you move DownsamplingThread elsewhere
from scipy.signal import decimate

# TODO: add some docs

# needed to put imported code on the path when running compiled version
# TODO: make sure that this does the trick... looks like original code is adding
    # subpaths of the OE plugin install dir
    # NOTE: might also just do what the old TODO (below) says...
# sys.path.append(dirname(__file__))
# sys.path.append(abspath(join(dirname(__file__),'../')))
# sys.path.append('/Users/fpbatta/src/GUImerge/GUI/Plugins')
# sys.path.append('/Users/fpbatta/src/GUImerge/GUI/Plugins/multiprocessing_plugin') # TODO put the python path in the C++ executalbe
sys.path.append('/Users/benacland/devdir/PythonPlugin/python_modules')
sys.path.append('/Users/benacland/devdir/PythonPlugin/python_modules/blp_trace') # TODO put the python path in the C++ executalbe

isDebug = False

# =======================================
# =           Abstract Plugin           =
# =======================================

class BaseMultiprocPlugin(object):
    def __init__(self):
        super(BaseMultiprocPlugin, self).__init__()
        self.ctrl_pipe = None
        self.controller = None
        self.ctrl_processes = None
        self.has_child = False

    def startup(self, sr):
        # we'll use the 'spawn' start method
        ctx = mp.get_context('spawn')
        
        # the OE plugin sets the 'PYTHONHOME' env variable... so we'll use that
        pyPath = join(os.environ.get('PYTHONHOME','dammit'), 'bin', 'python')
        if not (pyPath and isfile(pyPath)):
            raise FileNotFoundError("Could not find python executable '{}'".format(pyPath))
        ctx.set_executable(pyPath)

        # subclasses should override to set self.controller to a subclass of BasePlotController
        self.init_controller(int(sr))

        # start up the subprocess
        self.ctrl_pipe, controllers_pipe = ctx.Pipe()
        self.ctrl_processes = ctx.Process(target=self.controller, args=(controllers_pipe,))
        self.ctrl_processes.daemon = True
        self.ctrl_processes.start()
        self.has_child = True

    def is_ready(self):
        # TODO: (later) ask the subprocess to tell us if it's ready
        return self.has_child

    def param_config(self):
        return self.controller.param_config()

    def send_subproc_command(self, command, *args, **kwargs):
        self.ctrl_pipe.send(('cmd', command, args, kwargs))

    def send_subproc_param(self, name, value):
        self.send_subproc_command('set_param', name, value)

    def bufferfunction(self, n_arr=None, finished=False):
        # we pass all channels to the controller, and let the decision about
        # which to pay attention to fall on the controller's shoulders

        # update both of the following for the new pipe message rules
        if finished:
            print("asking the subprocess to stop...")
            self.send_subproc_command('terminate')
            # wait for the subprocess to close, close the pipe, then return immediately
            self.ctrl_processes.join()
            self.ctrl_pipe.close()
            return []
        else:
            self.ctrl_pipe.send({'data': n_arr})
        # NOTE: the following does not ensure that events related to the buffer
        # we just passed the subprocess will be returned on this call to
        # bufferfunction(). But that's probably ok... they'll be returned on a
        # subsequent call (hopefully).
        events = []
        while self.ctrl_pipe.poll():
            events.append(self.ctrl_pipe.recv())
        return events

    def __setattr__(self, key, value):
        # if the subproc class has a matching param config, it gets the call
        if hasattr(self,'controller') and self.controller is not None:
            # hasattr needed to prevent crash during init
            if key in self.controller.param_config():
                self.send_subproc_param(key, value)
                return
        # TODO: it might be nice to keep track of the param configs in this
        # class, and tie into the OE plugin param persistence machinery
        object.__setattr__(self, key, value)

    def __del__(self):
        self.bufferfunction(finished=True)

    # -----------  Virtual Methods  -----------

    def init_controller(self, input_frequency):
        """
        Subclasses should override this method to set self.controller to an
        instance of a concrete subclass of BasePlotController
        # TODO: eventually, this could be 'a concrete subclass of BaseController'
        """
        raise NotImplementedError()

    def plugin_name(self):
        """Subclasses should override to tell us their name
        
        Returns:
            string: The name of this plugin
        
        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError()

# ============================================
# =           Abstract Controllers           =
# ============================================

class BaseController(object):
    """
    Nothing, for now... but once we do more than plotting in subprocesses, this
    will take most of the contents of BasePlotController
    """
    def __init__(self):
        super(BaseController, self).__init__()

class BasePlotController(BaseController):
    def __init__(self, input_frequency, plot_frequency=0.1):
        self.pipe = None
        self._input_frequency = input_frequency
        self.plot_frequency = plot_frequency
        self.pipe_reader = None
        self.pipe_thread = None
        self.preprocessors = []
        self.preproc_threads = []
        self.plot_input_buffer = None
        self.params = {}
        # event that lets us know when to knock it off (initialized later)
        self.should_die = None

    def __call__(self, pipe):
        # NOTE: subclasses should not override this method, and should perform
        # all initialization actions in the various init_() methods. Plot
        # initialization should happen in start_plotting().
        
        # we don't set this in init because Event objects cannot be pickled
        self.should_die = Event()
        self.should_die.clear()
        
        # keep a ref to the pipe for sending objects back to the parent process
        self.pipe = pipe

        # set up the pipe reader
        self.init_pipe_reader()

        # set up preprocessors
        self.preprocessors = []
        self.init_preprocessors()

        # we'll call update_plot() under the lock belonging to the output buffer
        # of either the pipe cleaner or the last preprocessing step
        if self.preprocessors:
            self.plot_input_buffer = self.preprocessors[-1].output_buff
        else:
            self.plot_input_buffer = self.pipe_reader.buffer

        # start the preprocessing threads
        self.preproc_threads = []
        for pp in self.preprocessors:
            ppt = Thread(target=pp, daemon=True)
            self.preproc_threads.append(ppt)
            ppt.start()

        # start the pipe cleaner
        self.pipe_thread = Thread(target=self.pipe_reader, daemon=True)
        self.pipe_thread.start()

        # init the plot, and start plotting!
        self.start_plotting()

    def init_pipe_reader(self):
        """
        subclasses can override to customize the pipe_reader settings
        """
        self.pipe_reader = PipeCleaner(self.pipe,
            buff_len=int(self.input_frequency*20),  # 20s input buffer
            interval=0.001                          # pause 1ms between buffer checks
            )

    def is_ready(self):
        # TODO: this isn't used at this point... see method with same name in
        # base plugin class
        return 1
        
    def parse_command(self, cmd_tuple):
        # second item in the tuple is the name of the command.
        cmd = cmd_tuple[1]
        # optional third and fourth items can be args and kwargs for the command
        args = []
        kwargs = {}
        if len(cmd_tuple) > 2:
            for item in cmd_tuple[2:]:
                if isinstance(item, (list,tuple)):
                    args = item
                elif isinstance(item, (dict,)):
                    kwargs = item
        return (cmd, args, kwargs)

    def set_param(self, pKey, pVal):
        self.params[pKey] = pVal

    def gui_callback(self):
        # check whether you should continue
        if self.should_die.is_set():
            return

        # handle any messages that might have arrived
        commands = []
        other_msgs = [] # right now we don't do anything with these
        with self.pipe_reader.msg_lock:
            for msg in self.pipe_reader.read_messages():
                # right now we just look for commands and pass them to
                # self.handle_command
                if (msg[0] == 'cmd'):
                    commands.append(msg)
                else:
                    other_msgs.append(msg)
        # process commands in the order the were received
        for msg in commands:
            # parse the command
            cmd, args, kwargs = self.parse_command(msg)
            # handle the command
            self.handle_command(cmd, *args, **kwargs)
            # if the command was "terminate," don't process anything else
            if cmd == 'terminate':
                break

        # check whether you should continue
        if self.should_die.is_set():
            return

        # NOTE: handle other messages before acquiring the ui input buffer lock

        # check whether you should continue
        if self.should_die.is_set():
            return

        # get the ui input buffer lock
        isLocked = self.plot_input_buffer.rlock.acquire(blocking=True, timeout=1.0)
        if not isLocked:
            self.terminate(RuntimeError("Oh snap! Unable to acquire the ui input buffer lock..."))
            return

        # update the plot
        self.update_plot()

        # release the ui input buffer lock
        self.plot_input_buffer.rlock.release();

    def handle_command(self, command, *args, **kwargs):
        # make sure the command maps to a method this class implements
        if not (command in dir(self)):
            warn("no method found for command: '{}'".format(command))
            return # maybe we should print a message about this...
        m = getattr(self, command)
        # N.B. the following doesn't check the argspec, so craft your commands
        # carefully
            # TODO: a better job of this, using getargspec(m)
        if (args and kwargs):
            m(*args, **kwargs)
        elif args:
            m(*args)
        elif kwargs:
            m(**kwargs)
        else:
            m()

    def terminate(self, ex=None):
        # after setting this event, gui_callback() won't process anything else
        self.should_die.set()

        # kill the pipe reading thread and close the pipe
        self.pipe_reader.kill_thread()
        self.pipe_thread.join()
        self.pipe_reader = None
        self.pipe_thread = None
        self.pipe.close()
        self.pipe = None

        # kill the preprocessing threads
        for i in range(len(self.preprocessors)):
            self.preprocessors[i].kill_thread()
            self.preproc_threads[i].join()
        self.preprocessors = []
        self.preproc_threads = []

        # stop whatever UI callback the subclass set up
        self.stop_plotting()
        
        # If there was an exception, raise it now
        if ex:
            raise ex
        sys.exit(0)

    # -----------  Virtual Static Methods  -----------
    
    @staticmethod
    def param_config():
        # subclasses may override if they want to surface UI elements in the OE GUI
        return ()

    # -----------  Virtual Methods  -----------
    
    def start_plotting(self):
        """
        Subclasses should override this method.
        
        Overrides should do two things:
            1. Plot initialization
            2. Set up a timer that calls self.gui_callback() on a thread that
               can manipulate the UI (usually the main thread). Most gui kits
               offer something like this. For example, the instructions for
               setting up a timer using matplotlib can be found here:
               https://matplotlib.org/examples/event_handling/timers.html
        
        Raises: NotImplementedError: If you don't override this method, we'll
        complain.
        """
        raise NotImplementedError("subclasses should override start_plotting()")

    def stop_plotting(self):
        """
        Subclasses should override to kill whatever timer or thread they set up
        in start_plotting(), close the plot
        
        This function is called from self.terminate(), and should return once
        the timer is dead.
        
        Raises: NotImplementedError: If you don't override thie method, we'll complain.
        """
        raise NotImplementedError("subclasses should override stop_plotting()")
    
    def update_plot(self):
        """
        update your UI in this method, called from self.gui_callback()
        
        Note that this call is made under the plot input buffer's lock... so
        make your plot update snappy!
        
        Raises: NotImplementedError: If you don't override this method, we'll
        complain.
        
        @param      events  Any events not handled automatically by the base
                            class. Do with them what you will.
        """
        raise NotImplementedError("subclasses should override update_plot()")

    def init_preprocessors(self):
        """
        subclasses can override to set self.preprocessors to an ordered list of
        BasePreprocThread subclasses
        """
        raise NotImplementedError("subclasses should override init_preprocessors()")

    # -----------  Read-Only Properties  -----------

    @property
    def input_frequency(self):
        return self._input_frequency
    @input_frequency.setter
    def input_frequency(self, val):
        pass

    # -----------  Unimplemented  -----------

    def handleEvents(eventType,sourceID,subProcessorIdx,timestamp,sourceIndex):
        """handle events passed from OE"""
        pass
    def handleSpike(self, electrode, sortedID, n_arr):
        """handle spikes passed from OE"""
        pass
    

# =============================================
# =           Preprocessing Threads           =
# =============================================

class BasePreprocThread(object):
    def __init__(self, input_buff, output_dtype=np.float64, output_buff_len=30000*20, interval=0.01):
        super(BasePreprocThread, self).__init__()
        self.interval = interval
        # set up the input and output buffers
        self._input_buff = input_buff
        self._output_buff = CircularBuff(output_dtype, output_buff_len)
        # event that lets us know when to knock it off
        self.should_die = Event()
        self.should_die.clear()

    def __call__(self):
        while True:
            # make sure we should keep going
            if self.should_die.is_set():
                break
            # now go do that voodoo that you do so well
            self.process()
            # make sure we should still keep going
            if self.should_die.is_set():
                break
            # wait for a bit before continuing. NOTE: there will be some drift in
            # loop start times... but that shouldn't be a problem.
            sleep(self.interval)

    def kill_thread(self):
        # after setting this event, the thread will die before the next (or
        # after an ongoing) call to self.process()
        self.should_die.set()

    # -----------  Virtual Methods  -----------

    def process(self):
        raise NotImplementedError("subclasses of BasePreprocThread should override the method process()")

    # -----------  Read-Only Properties  -----------

    @property
    def input_buff(self):
        return self._input_buff
    @input_buff.setter
    def input_buff(self, val):
        pass

    @property
    def output_buff(self):
        return self._output_buff
    @output_buff.setter
    def output_buff(self, val):
        pass

class DownsamplingThread(BasePreprocThread):
    """Decimates data from one buffer, puts it in another"""
    def __init__(self, fsIn, fsOut, chunk_size=None, *args, **kwargs):
        """Init function.
        
        Args:
            fsIn (int): Frequency of incoming data. Must be a multiple of fsOut.
            fsOut (int): Frequency of data after downsampling.
            chunk_size (int, optional): Data will be downsampled in multiples of chunk_size.
            *args: positional arguments to BasePreprocThread initializer (just the input_buff)
            **kwargs: keyword arguments to BasePreprocThread initializer
        """
        super(DownsamplingThread, self).__init__(*args, **kwargs)
        self._fsIn = int(fsIn)
        self._fsOut = int(fsOut)
        self._ratio = int(fsIn/fsOut)
        if not (chunk_size is None):
            self._chunk_size = int(chunk_size)
        else:
            self._chunk_size = int(4*fsIn/fsOut)

    # overrides

    def process(self):
        # we'll do our work with the input buffer inside of its lock
        with self.input_buff.rlock:
            nChunks = int(np.floor(self.input_buff.nUnread / self.chunk_size))
            if nChunks == 0: # no complete chunks? nothing to do!
                return;
            # retrieve a copy of the input, then do the rest outside of the lock
            d = self.input_buff.read(nChunks * self.chunk_size)
        
        # decimate as many complete chunks as you can (along the last axis)
        self.output_buff.write(decimate(d, self.ratio, axis=d.ndim-1, zero_phase=True))

    # -----------  Read-Only Properties  -----------

    @property
    def fsIn(self):
        return self._fsIn
    @fsIn.setter
    def fsIn(self, val):
        pass

    @property
    def fsOut(self):
        return self._fsOut
    @fsOut.setter
    def fsOut(self, val):
        pass

    @property
    def ratio(self):
        return self._ratio
    @ratio.setter
    def ratio(self, val):
        pass

    @property
    def chunk_size(self):
        return self._chunk_size
    @chunk_size.setter
    def chunk_size(self, val):
        pass

# ===========================
# =           IPC           =
# ===========================

class PipeCleaner(object):
    """Copies input from a pipe to a buffer and message queue"""
    def __init__(self, pipe, dtype=np.float64, buff_len=30000*20, interval=0.001, msg_lock=RLock(), buff_lock=RLock()):
        super(PipeCleaner, self).__init__()
        self.pipe = pipe
        self._buffer = CircularBuff(dtype, buff_len, rlock=buff_lock)
        self._msg_lock = msg_lock
        self._msg_queue = deque()
        self.interval = interval
        # event that lets us know when to knock it off
        self.should_die = Event()
        self.should_die.clear()

    def __call__(self):
        while True:
            # make sure we should keep going
            if self.should_die.is_set():
                break
            # grab everything that's currently in the pipe
            self.clear_pipe()
            # make sure we should still keep going
            if self.should_die.is_set():
                break
            # wait for a bit before continuing. NOTE: there will be some drift in
            # loop start times... but that shouldn't be a problem.
            sleep(self.interval)

    def clear_pipe(self):
        """
        Moves all data and messages into the data buffer and message queue
        
        Pipe contents should all be tuples, with the first entry specifying how
        the rest of the tuple should be handled type. Supported formats are:
            1. data: ('data', numpy.ndarray((nChans,X), self.dtype)) where type(X) == int
            2. commands: ('cmd', cmd_name, [args(array/tuple)], [kwargs(dict)])
                # N.B. either or both of args and kwargs can be omitted
        """
        new_msgs = []
        while self.pipe.poll():
            thing = self.pipe.recv()
            if not (type(thing) == tuple):
                continue
            if thing[0] == 'data':
                self.buffer.write(thing[1]);
            else:
                new_msgs.append(thing)
        # add new events to the queue (using the msg_lock)
        if new_msgs:
            with self.msg_lock:
                self.msg_queue.extend(new_msgs)
    
    def read_messages(self):
        # with msg_lock, copy all events from the msg_queue into an array
        with self.msg_lock:
            n = len(self.msg_queue)
            return [self.msg_queue.popleft() for x in range(n)]

    def kill_thread(self):
        # after setting this event, the thread will die before the next (or
        # after an ongoing) call to self.clear_pipe()
        self.should_die.set()
    
    # -----------  Read-Only Properties  -----------

    @property
    def buffer(self):
        return self._buffer
    @buffer.setter
    def buffer(self, val):
        pass

    @property
    def msg_lock(self):
        return self._msg_lock
    @msg_lock.setter
    def msg_lock(self, val):
        pass

    @property
    def msg_queue(self):
        return self._msg_queue
    @msg_queue.setter
    def msg_queue(self, val):
        pass

# ===========================================
# =           Thread-Safe Buffers           =
# ===========================================

class CircularBuff(object):
    """
    A basic thread-safe circular buffer.
    
    To acquire the reenrtant lock used for array opperations, use the rlock
    property. Make the buffer's long enough that it won't wrap in whatever
    conditions apply... otherwise behavior is undefined.

    # TODO: push rIdx forward if wIdx catches up

    # TODO: fail noisily.
    """
    def __init__(self, dtype, length, rlock=RLock()):
        """Initializes the buffer and sets some read-only properties
        
        Args:
            dtype (type): should be numeric, but can be any type that works with ndarray
            length (int): number of samples (columns) the buffer can store
            rlock (RLock, optional): The RLock to use when working on the buffer
        """
        super(CircularBuff, self).__init__()
        self._dtype = dtype
        self._length = length
        self._rIdx = 0
        self._wIdx = 0
        self._rlock = rlock
        # initialize the buffer to None at first... it'll be created on the
        # first call to write()
        self._buffer = None

    # reading and writing

    def read(self, nSamps):
        """
        Reads nSamps from the buffer (for all channels). then removes them from
        the buffer.
        
        Args: nSamps (int): The number of samples to read
        
        Returns: ndarray((nChans,nSamps) self.dtype): nChans may change...
        
        Raises: IndexError: If you ask for more samples than are available,
        we'll complain.
        """
        with self.rlock:
            # NOTE: if you don't have a buffer, return None. That's what the
            # user gets for not asking if there's anything to be read!
            if self.buffer is None:
                return None

            # If the user asks for too much... give them a pie in the face
            if nSamps > self.nUnread:
                raise IndexError("Asked for {}, but I only have {}".format(nSamps,self.nUnread))

            # Ok, it's a reasonable request. Let's give them what they asked for.
            out = np.ndarray((np.shape(self.buffer)[0], nSamps), self.dtype)
            if self.rIdx + nSamps <= self.length:
                out[:,:] = self.buffer[:,self.rIdx:self.rIdx+nSamps]
            else:
                r = self.length - self.rIdx
                out[:,0:r] = self.buffer[:,self.rIdx:]
                out[:,r:] = self.buffer[:,0:(nSamps-r)]
            self._rIdx = (self._rIdx + nSamps) % self.length

            # if there's only 1 channel, squeeze the output
            return np.squeeze(out) if np.shape(out)[0]==1 else out

    def write(self, samps):
        """Copies 'samps' to the buffer..
        
        NOTE: The buffer will automatically resize if the number of input rows
              differs from previous calls.
        
        Args:
            samps (np.ndarray): the data to add to the buffer
        """
        # check for empty
        if len(samps)==0:
            return
        with self.rlock:
            # create a 2d view of samps so we can index 1d and 2d input the same way
            inShape = np.shape(samps)
            inChans = 1 if len(inShape) == 1 else inShape[0]
            inSamps = inShape[-1]
            samps = samps.reshape((inChans, inSamps))
            
            # if you don't have a buffer yet, create one.
            if self._buffer is None:
                self._buffer = np.zeros((inChans, self.length), self.dtype)

            # if the number of channels in the input doesn't match your buffer,
            # resize the buffer.
            if inChans != np.shape(self.buffer)[0]:
                self._buffer.resize((inChans, self.length), refcheck=False)

            # copy samps into the buffer
            if self.wIdx + inSamps <= self.length:
                self.buffer[:,self.wIdx:(self.wIdx+inSamps)] = samps
            else:
                r = self.length - self.wIdx
                self.buffer[:,self.wIdx:] = samps[:,0:r]
                self.buffer[:,0:(inSamps-r)] = samps[:,r:]
            self._wIdx = (self._wIdx + inSamps) % self.length

    # -----------  Read-Only Properties  -----------

    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self, val):
        pass

    @property
    def length(self):
        return self._length
    @length.setter
    def length(self, val):
        pass

    @property
    def rlock(self):
        return self._rlock
    @rlock.setter
    def rlock(self, val):
        pass

    @property
    def buffer(self):
        return self._buffer
    @buffer.setter
    def buffer(self, val):
        pass

    @property
    def rIdx(self):
        return self._rIdx
    @rIdx.setter
    def rIdx(self, val):
        pass

    @property
    def wIdx(self):
        return self._wIdx
    @wIdx.setter
    def wIdx(self, val):
        pass

    @property
    def nUnread(self):
        with self.rlock:
            return (self._wIdx - self._rIdx) % self.length
    @nUnread.setter
    def nUnread(self, val):
        pass

class ConstantLengthCircularBuff(CircularBuff):
    """
    Like CircularBuff, but rIdx is always wIdx+1.
    
    When you call read(), you'll always get back a buffer with self.length cols.
    """
    def __init__(self, *args, **kwargs):
        super(ConstantLengthCircularBuff, self).__init__(*args, **kwargs)
        self._wIdx = 0
        self._rIdx = 1
    
    def write(self, samps):
        # call super, then update rIdx
        with self.rlock:
            super(ConstantLengthCircularBuff, self).write(samps)
            self.rIdx = (self.wIdx+1) % self.length

    def read(self):
        with self.rlock:
            return np.hstack([self.buffer[:,self.rIdx:], self.buffer[:,:self.rIdx]])

