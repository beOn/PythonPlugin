from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
        name= "blp_trace",
        ext_modules = cythonize(Extension('blp_trace',sources=["blp_trace.pyx"],export_symbols=['pluginStartup','pluginisready','getParamNum','getParamConfig','pluginFunction','eventFunction','spikeFunction','setIntParam','setFloatParam','getIntParam','getFloatParam'])),
        include_dirs = [numpy.get_include()]
        )
