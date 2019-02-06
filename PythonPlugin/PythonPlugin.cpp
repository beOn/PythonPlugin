/*
 ------------------------------------------------------------------
 
 Python Plugin
 Copyright (C) 2016 FP Battaglia
 
 based on
 Open Ephys GUI
 Copyright (C) 2013, 2015 Open Ephys
 
 ------------------------------------------------------------------
v 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
 */
/*
  ==============================================================================

    PythonPlugin.cpp
    Created: 13 Jun 2014 5:56:17pm
    Author:  fpbatta

  ==============================================================================
*/

#include "PythonPlugin.h"
#include "PythonEditor.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <stdlib.h>
#include <string.h>

#ifdef DEBUG
#define PYTHON_DEBUG TRUE
#endif

#ifdef PYTHON_DEBUG
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#elif !defined(_WIN32)
#include <pthread.h> 
#endif
#endif


PythonPlugin::PythonPlugin(const String &processorName)
    : GenericProcessor(processorName) //, threshold(200.0), state(true)

{

    //parameters.add(Parameter("thresh", 0.0, 500.0, 200.0, 0));
    filePath = "";
    plugin = 0;

    char * old_python_home = getenv("PYTHONHOME");
    if (old_python_home == NULL)
    {
#ifdef PYTHON_DEBUG
        std::cout << "setting PYTHONHOME" << std::endl;
#endif

#ifdef _WIN32
        _putenv_s("PYTHONHOME", "C:\\Users\\Ephys\\Anaconda3"); // set to default PYTHONHOME by PythonEnv.props
#else
#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)
#define PYTHON_HOME_NAME STR(PYTHON_HOME)

        //setenv("PYTHONHOME", "/anaconda3/bin/", 1); // FIXME hardcoded PYTHONHOME!
        setenv("PYTHONHOME", PYTHON_HOME_NAME, 1);
        //setenv("PYTHONHOME", "/anaconda3/bin/python.app", 1); // FIXME hardcoded PYTHONHOME!
#endif
    }
    // setenv("PYTHONHOME", "/usr/local/anaconda", 1); // FIXME hardcoded PYTHONHOME!

#ifdef PYTHON_DEBUG
    std::cout << "PYTHONHOME: " << getenv("PYTHONHOME") << std::endl;
#endif
    
#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#elif defined(_WIN32)
    DWORD tid = GetCurrentThreadId();
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    std::cout << "in constructor pthread_threadid_np()=" << tid << std::endl;
#endif

#if PY_MAJOR_VERSION==3
    Py_SetProgramName ((wchar_t *)"PythonPlugin");
#else
    Py_SetProgramName ((char *)"PythonPlugin");
#endif
    //Py_SetPythonHome("Users/ClaytonBarnes/Anaconda3");
    //Py_SetPath("Users/ClaytonBarnes/Anaconda3/lib");
    Py_Initialize ();
    PyEval_InitThreads();

    
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.setcheckinterval(10000)");
#ifdef PYTHON_DEBUG
    std::cout << Py_GetPrefix() << std::endl;
    std::cout << Py_GetVersion() << std::endl;
#endif
    GUIThreadState = PyEval_SaveThread();
}

PythonPlugin::~PythonPlugin()
{
#ifdef _WIN32
	//Close libary
	PyGILState_Ensure();
	FreeLibrary((HMODULE)plugin);
#else
	dlclose(plugin);
#endif
}

void PythonPlugin::createEventChannels()
{
    EventChannel* ev = new EventChannel(EventChannel::TTL, 8, 1,  CoreServices::getGlobalSampleRate(), this);
    ev->setName("Python events");
    ev->setDescription("Events generated by a Python plugin");
    String identifier = "dataderived.python";
    ev->setIdentifier(identifier);
    MetaDataDescriptor md(MetaDataDescriptor::CHAR, 25, "Plugin type", "Description of the plugin", "channelInfo.extra");
    MetaDataValue mv(md);
    mv.setValue("Python plugin");
    ev->addMetaData(md, mv);
    eventChannelArray.add(ev);
    ttlChannel = ev;
    std::cout << "Python event channel created" << std::endl;

}

AudioProcessorEditor* PythonPlugin::createEditor()
{

//        std::cout << "in PythonEditor::createEditor()" << std::endl;
    editor = new PythonEditor(this, true);

    return editor;

}


bool PythonPlugin::isReady()
{
#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#elif defined(_WIN32)
    DWORD tid = GetCurrentThreadId();
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    std::cout << "in isReady pthread_threadid_np()=" << tid << std::endl;
#endif

    bool ret;
    PyEval_RestoreThread(GUIThreadState);
    if (plugin == 0 )
    {
        CoreServices::sendStatusMessage ("No plugin selected in Python Plugin.");
        ret = false;
    }
    else if (pluginIsReady && !(*pluginIsReady)())
    {
        CoreServices::sendStatusMessage ("Python Plugin is not ready");
        ret = false;
    }
    else
    {
        ret = true;
    }
    GUIThreadState = PyEval_SaveThread();
    return ret;

}

void PythonPlugin::setParameter(int parameterIndex, float newValue)
{
    editor->updateParameterButtons(parameterIndex);

    //Parameter& p =  parameters.getReference(parameterIndex);
    //p.setValue(newValue, 0);

    //threshold = newValue;

    //std::cout << float(p[0]) << std::endl;
    editor->updateParameterButtons(parameterIndex);
}


void PythonPlugin::resetConnections()
{
#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#elif defined(_WIN32)
    DWORD tid = GetCurrentThreadId();
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    std::cout << "in resetConnection pthread_threadid_np()=" << tid << std::endl;
#endif

    nextAvailableChannel = 0;
    
    wasConnected = false;

#ifdef PYTHON_DEBUG
    std::cout << "resetting ThreadState, which was "  << processThreadState << std::endl;
#endif
    processThreadState = 0;
}

void PythonPlugin::process(AudioSampleBuffer& buffer)
{
    checkForEvents(true);

#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#elif defined(_WIN32)
    DWORD tid = GetCurrentThreadId();
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    // std::cout << "in process pthread_threadid_np()=" << tid << std::endl;
#endif
    
    
    if(!processThreadState)
    {
        
        //DEBUG
        PyThreadState *nowState;
        nowState = PyGILState_GetThisThreadState();
#ifdef PYTHON_DEBUG
        std::cout << "currentState: " << nowState << std::endl;
        std::cout << "initialiting ThreadState" << std::endl;
#endif
        if(nowState) //UGLY HACK!!!
        {
            processThreadState = nowState;
        }
        else
        {
            processThreadState =  PyThreadState_New(GUIThreadState->interp);
        }
        if(!processThreadState)
            std::cout << "ThreadState is Null!" << std::endl;
    }

    PyEval_RestoreThread(processThreadState);
    
    PythonEvent *pyEvents = (PythonEvent *)calloc(1, sizeof(PythonEvent));
    pyEvents->type = 0; // this marks an empty event
#ifdef PYTHON_DEBUG
    // std::cout << "in process, trying to acquire lock" << std::endl;
#endif 
    
    // PyEval_InitThreads();
//    
//    std::cout << "in process, threadstate: " << PyGILState_GetThisThreadState() << std::endl;
//    PyGILState_STATE gstate;
//    gstate = PyGILState_Ensure();
//    std::cout << "in process, lock acquired" << std::endl;
    (*pluginFunction)(*(buffer.getArrayOfWritePointers()), buffer.getNumChannels(), buffer.getNumSamples(), getNumSamples(0), pyEvents);
//    PyGILState_Release(gstate);
//    std::cout << "in process, lock released" << std::endl;
    
    if(wasTriggered)
    {
        uint8 ttlData = 0;
        // std::cout << "in Python plugin resetting channel: " << lastChan << std::endl;
        TTLEventPtr event = TTLEvent::createTTLEvent(ttlChannel, getTimestamp(0),
                                                     &ttlData, sizeof(uint8), lastChan);
        addEvent(ttlChannel, event, 0);
        wasTriggered = false;
    }
    if(pyEvents->type != 0)
    {
#ifdef PYTHON_DEBUG
        // std::cout << "Event emitted " << (int)pyEvents->type << std::endl;
#endif
        //            uint8 ttlData = 1 << module.outputChan;
        //            TTLEventPtr event = TTLEvent::createTTLEvent(moduleEventChannels[m], getTimestamp(module.inputChan) + i, &ttlData, sizeof(uint8), module.outputChan);
        //            addEvent(moduleEventChannels[m], event, i);
        lastChan = (uint16)pyEvents->eventId;
        uint8 ttlData = 1 << lastChan;
        
        // std::cout << "in Python plugin ts is " << getTimestamp(0) + pyEvents->sampleNum << " and sampleNum is " <<
        // pyEvents->sampleNum << " eventId: " <<  uint16(pyEvents->eventId) <<  " ttlData: " << int(ttlData) << std::endl;
        // FIXME now we set the ts at the first samble in the block
        TTLEventPtr event = TTLEvent::createTTLEvent(ttlChannel, getTimestamp(0) + pyEvents->sampleNum,
                                                     &ttlData, sizeof(uint8), lastChan);
        addEvent(ttlChannel, event, pyEvents->sampleNum);
        
        PythonEvent *lastEvent = pyEvents;
        PythonEvent *nextEvent = lastEvent->nextEvent;
        free((void *)lastEvent);
        wasTriggered = true;
        
        // std::cout << "lastChan is " << lastChan << std::endl;
        while (nextEvent) {
            lastChan = (uint16)nextEvent->eventId;
            uint8 ttlData = 1 << lastChan;
            // std::cout << "in Python plugin ts is " << getTimestamp(0) << " and sampleNum is " <<
            // nextEvent->sampleNum << " ttlData: " << ttlData << std::endl;

            TTLEventPtr event = TTLEvent::createTTLEvent(ttlChannel, getTimestamp(0) + nextEvent->sampleNum,
                                                         &ttlData, sizeof(uint8), lastChan);
            addEvent(ttlChannel, event, nextEvent->sampleNum);
            lastEvent = nextEvent;
            nextEvent = nextEvent->nextEvent;
            free((void *)lastEvent);
            
            wasTriggered = true;
        }
    }
    
    processThreadState = PyEval_SaveThread();
#ifdef PYTHON_DEBUG
    // std::cout << "Thread saved" << std::endl;
#endif
}

/** START CJB ADDED **/

void PythonPlugin::handleEvent(const EventChannel* eventInfo, const MidiMessage& event, int sampleNum){
    /** For reference
     in event info
     uint16 getCurrentNodeID() const;
     //Gets the index of this channel in the processor which currently owns this copy of the info object
     uint16 getCurrentNodeChannelIdx() const;
     // Gets the type of the processor which currently owns this copy of the info object
     String getCurrentNodeType() const;
     // Gets the name of the processor which currently owns this copy of the info object
     String getCurrentNodeName() const;
     
     # struct PythonEvent:
     # unsigned char type
     # int sampleNum
     # unsigned char eventId
     # unsigned char eventChannel
     # unsigned char numBytes
     # unsigned char *eventData
     # PythonEvent *nextEvent
     **/
    
    /**
     
     enum EventChannelTypes
     {
     //Numeration kept to maintain compatibility with old code
     TTL = 3,
     TEXT = 5,
     //generic binary types. These will be treated by the majority of record engines as simple binary blobs,
     //while having strict typing helps creating stabler plugins
     INT8_ARRAY = 10,
     UINT8_ARRAY,
     INT16_ARRAY,
     UINT16_ARRAY,
     INT32_ARRAY,
     UINT32_ARRAY,
     INT64_ARRAY,
     UINT64_ARRAY,
     FLOAT_ARRAY,
     DOUBLE_ARRAY,
     //For error checking
     INVALID,
     //Alias for checking binary types
     BINARY_BASE_VALUE = 10
     };
     
     **/
    
    /**
     
     #ifdef PYTHON_DEBUG
     #if defined(__linux__)
     pid_t tid;
     tid = syscall(SYS_gettid);
     #else
     uint64_t tid;
     pthread_threadid_np(NULL, &tid);
     #endif
     std::cout << "in setfloatparam pthread_threadid_np()=" << tid << std::endl;
     #endif
     PyEval_RestoreThread(GUIThreadState);
     (*setFloatParamFunction)(name.getCharPointer().getAddress(), value);
     GUIThreadState = PyEval_SaveThread();
     **/
    
    /**
     
     Event packet structure:
     EventType - 1byte
     SubType - 1byte
     Source processor ID - 2bytes
     Source Subprocessor index - 2 bytes
     Source Event index - 2 bytes
     Timestamp - 8 bytes
     Event Virtual Channel - 2 bytes
     data - variable
     
     
     EventChannel::EventChannelTypes getEventType() const;
     const EventChannel* getChannelInfo() const;
     uint16 getChannel() const;
     const void* getRawDataPointer() const;
     
     static EventChannel::EventChannelTypes getEventType(const MidiMessage& msg);
     
     **/
    int eventType;
    int sourceID;
    int subProcessorIdx;
    double timestamp;
    int sourceIndex;
    const void* ptr;
    
    if (eventInfo->getChannelType() == EventChannel::TTL)
    {
        
        TTLEventPtr ttl = TTLEvent::deserializeFromMessage(event, eventInfo);
        
        eventType = int(ttl->getEventType());
        sourceID = int(ttl->getSourceID());
        subProcessorIdx = int(ttl->getSubProcessorIdx());
        timestamp = double(ttl->getTimestamp());
        sourceIndex = int(ttl->getSourceIndex());
        //ptr = ttl->getRawDataPointer();
        sendEventPlugin(eventType, sourceID, subProcessorIdx, timestamp, sourceIndex);
    }
    else if (eventInfo->getChannelType() == EventChannel::TEXT)
    {
        
        TextEventPtr txt = TextEvent::deserializeFromMessage(event, eventInfo);
        eventType = int(txt->getEventType());
        sourceID = int(txt->getSourceID());
        subProcessorIdx = int(txt->getSubProcessorIdx());
        timestamp = double(txt->getTimestamp());
        sourceIndex = int(txt->getSourceIndex());
        ptr = txt->getRawDataPointer();
        sendEventPlugin(eventType, sourceID, subProcessorIdx, timestamp, sourceIndex);
    }
    else if (eventInfo->getChannelType() == EventChannel::TEXT)
    {
        
        BinaryEventPtr bi = BinaryEvent::deserializeFromMessage(event, eventInfo);
        eventType = int(bi->getEventType());
        sourceID = int(bi->getSourceID());
        subProcessorIdx = int(bi->getSubProcessorIdx());
        timestamp = double(bi->getTimestamp());
        sourceIndex = int(bi->getSourceIndex());
        //ptr = bi->getRawDataPointer();
        sendEventPlugin(eventType, sourceID, subProcessorIdx, timestamp, sourceIndex);
    }
}

void PythonPlugin::sendEventPlugin(int eventType, int sourceID, int subProcessorIdx, double timestamp, int sourceIndex){
#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#elif defined(_WIN32)
    DWORD tid = GetCurrentThreadId();
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    std::cout << "in sendEventPlugin pthread_threadid_np()=" << tid << std::endl;
#endif
    
    PyEval_RestoreThread(GUIThreadState);
    (*eventFunction)(eventType, sourceID, subProcessorIdx,timestamp,sourceIndex);
    GUIThreadState = PyEval_SaveThread();
}

void PythonPlugin::handleSpike(const SpikeChannel* spikeInfo, const MidiMessage& event, int samplePosition){
    /**
     const SpikeChannel* getChannelInfo() const;
     
     const float* getDataPointer() const;
     
     const float* getDataPointer(int channel) const;
     
     float getThreshold(int chan) const;
     
     uint16 getSortedID() const;
     
     
     
     
     
     **/
    
    SpikeEventPtr newSpike = SpikeEvent::deserializeFromMessage(event, spikeInfo);
    const float* dataPtr = newSpike->getDataPointer();
    float spikeBuf[18];
    for(int i = 0 ;i < 18;i++){
        spikeBuf[i] = dataPtr[i];
    }
    //juce::uint16
    int sortedID = int(newSpike->getSortedID());
    

#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#elif defined(_WIN32)
    DWORD tid = GetCurrentThreadId();
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    std::cout << "in handleSpike pthread_threadid_np()=" << tid << std::endl;
#endif
     
    /*
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    
    Perform Python actions here.
    result = CallSomeFunction();
    evaluate result or handle exception
    Release the thread. No Python API allowed beyond this point.
    PyGILState_Release(gstate);
    */
    //PyGILState_STATE gstate;
    //gstate = PyGILState_Ensure();
    
    if(!processThreadState)
    {
        //*
        
        //DEBUG
        //PyEval_RestoreThread(processThreadState);
        PyThreadState *nowState;
        nowState = PyGILState_GetThisThreadState();
#ifdef PYTHON_DEBUG
        std::cout << "currentState: " << nowState << std::endl;
        std::cout << "initialiting ThreadState" << std::endl;
#endif
        if(nowState) //UGLY HACK!!!
        {
            processThreadState = nowState;
        }
        else
        {
            processThreadState =  PyThreadState_New(GUIThreadState->interp);
        }
        if(!processThreadState)
            std::cout << "ThreadState is Null!" << std::endl;
    }
    
    PyEval_RestoreThread(processThreadState);
    
    PythonEvent *pyEvents = (PythonEvent *)calloc(1, sizeof(PythonEvent));
    
    pyEvents->type = 0; // this marks an empty event
#ifdef PYTHON_DEBUG
    // std::cout << "in process, trying to acquire lock" << std::endl;
#endif
    
     PyEval_InitThreads();
    //
    //    std::cout << "in process, threadstate: " << PyGILState_GetThisThreadState() << std::endl;
    //    PyGILState_STATE gstate;
    //    gstate = PyGILState_Ensure();
    //    std::cout << "in process, lock acquired" << std::endl;
    
    (*spikeFunction)(sortedID, spikeBuf);
    processThreadState = PyEval_SaveThread();

    //PyGILState_Release(gstate);
    
    //processThreadState = PyEval_SaveThread();
    /**
     #ifdef PYTHON_DEBUG
     #if defined(__linux__)
     pid_t tid;
     tid = syscall(SYS_gettid);
     #else
     uint64_t tid;
     pthread_threadid_np(NULL, &tid);
     #endif
     std::cout << "in handleSpike pthread_threadid_np()=" << tid << std::endl;
     #endif
     
     PyEval_RestoreThread(GUIThreadState);
     (*spikeFunction)(sortedID, spikeBuf);
     GUIThreadState = PyEval_SaveThread();
     **/
}

/** END CJB ADDED **/

/* The complete API that the Cython plugin has to expose is
 void pluginStartup(void): a function to initialize the plugin data structures prior to start ACQ
 int isReady(void): a boolean function telling the processor whether the plugin is ready  to receive data
 int getParamNum(void) get the number of parameters that the plugin takes
 ParamConfig *getParamConfig(void) this will allow generating the editor GUI TODO
 void setIntParameter(char *name, int value) set integer parameter
 void set FloatParameter(char *name, float value) set float parameter
 
 */

std::string GetLastErrorAsString()
{
    /*Get the error message, if any.*/
    DWORD errorMessageID = ::GetLastError();
    if(errorMessageID == 0)
        return std::string(); //No error message has been recorded

    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

    std::string message(messageBuffer, size);

    //Free the buffer.
    LocalFree(messageBuffer);

    return message;
}

void PythonPlugin::setFile(String fullpath)
{
#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#elif defined(_WIN32)
    DWORD tid = GetCurrentThreadId();
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    std::cout << "in setFile pthread_threadid_np()=" << tid << std::endl;
#endif
    
#ifdef _WIN32
	//Load plugin
    filePath = fullpath;
	std::string path = filePath.toStdString();
    plugin = LoadLibraryA(path.c_str());
#else
	filePath = fullpath;

	const char* path = filePath.getCharPointer();
	plugin = dlopen(path, RTLD_LAZY);
#endif
    if (!plugin)
      {
		  std::cout << "Can't open plugin "
			  << '"' << path << "\""	  
#ifdef _WIN32
              << GetLastErrorAsString()
#else
              << dlerror()	  
#endif
              << std::endl;
		  return;

      }

    //String initPlugin = filePath.fromLastOccurrenceOf(String("/"), false, true);
    //Path is dif in windows..
    String initPlugin = filePath.fromLastOccurrenceOf(String("\\"), false, true);
    
    initPlugin = initPlugin.upToFirstOccurrenceOf(String("."), false, true);
    
#if PY_MAJOR_VERSION>=3
    String initPluginName = String("PyInit_");
#else
    String initPluginName = String("init");
#endif
    initPluginName.append(initPlugin, 200);
    
    std::cout << "init function is: " << initPluginName << std::endl;
    
    void *initializer;
    
#ifdef _WIN32
	initializer = GetProcAddress((HMODULE)plugin, initPluginName.getCharPointer());
#else
	initializer = dlsym(plugin, initPluginName.getCharPointer());
#endif

#ifdef PYTHON_DEBUG
	std::cout << "initializer: " << initializer << std::endl;
#endif
    if (!initializer)
    {
        std::cout << "Can't find init function in plugin "
            << '"' << path << "\"" << std::endl
#ifdef _WIN32
            << GetLastErrorAsString()
#else
            << dlerror()
#endif
            << std::endl;
        plugin = 0;
        return;
    }

    initfunc_t initF = (initfunc_t) initializer;
	void *cfunc;
#ifdef _WIN32
    cfunc = GetProcAddress((HMODULE)plugin, "pluginisready");
#else
	cfunc = dlsym(plugin, "pluginisready");
#endif
    if (!cfunc)
    {
		std::cout << "Can't find ready function in plugin "
			<< '"' << path << "\"" << std::endl
#ifdef _WIN32
            << GetLastErrorAsString()
#else
            << dlerror()
#endif
			<< std::endl;
		plugin = 0;
		return;
    }
    pluginIsReady = (isreadyfunc_t)cfunc;

#ifdef _WIN32
	cfunc = GetProcAddress((HMODULE)plugin, "pluginStartup");
#else
	cfunc = dlsym(plugin, "pluginStartup");
#endif
    if (!cfunc)
    {
		std::cout << "Can't find startup function in plugin "
			<< '"' << path << "\"" << std::endl
#ifdef _WIN32
            << GetLastErrorAsString()
#else
            << dlerror()	  
#endif
			<< std::endl;
		plugin = 0;
		return;
    }
    pluginStartupFunction = (startupfunc_t)cfunc;
    
#ifdef _WIN32
	cfunc = GetProcAddress((HMODULE)plugin, "getParamNum");
#else
	cfunc = dlsym(plugin, "getParamNum");
#endif

    if (!cfunc)
    {
		std::cout << "Can't find getParamNum function in plugin "
			<< '"' << path << "\"" << std::endl
#ifdef _WIN32
            << GetLastErrorAsString()
#else
            << dlerror()	  
#endif
			<< std::endl;
		plugin = 0;
		return;
    }
    getParamNumFunction = (getparamnumfunc_t)cfunc;
    

#ifdef _WIN32
	cfunc = GetProcAddress((HMODULE)plugin, "getParamConfig");
#else
	cfunc = dlsym(plugin, "getParamConfig");
#endif
    if (!cfunc)
    {
		std::cout << "Can't find getParamNum function in plugin "
			<< '"' << path << "\"" << std::endl
#ifdef _WIN32
            << GetLastErrorAsString()
#else
            << dlerror()
#endif
			<< std::endl;
		   	plugin = 0;
		   	return;

    }
    getParamConfigFunction = (getparamconfigfunc_t)cfunc;

    
#ifdef _WIN32
	cfunc = GetProcAddress((HMODULE)plugin, "pluginFunction");
#else
	cfunc = dlsym(plugin, "pluginFunction");
#endif
    // std::cout << "plugin:   " << cfunc << std::endl;
    if (!cfunc)
    {
		std::cout << "Can't find plugin function in plugin "
			<< '"' << path << "\"" << std::endl
#ifdef _WIN32
            << GetLastErrorAsString()
#else
            << dlerror()	  
#endif
			<< std::endl;
		plugin = 0;
		return;
    }
    pluginFunction = (pluginfunc_t)cfunc;
    
    // CJB added start
#ifdef _WIN32
	cfunc = GetProcAddress((HMODULE)plugin, "eventFunction");
#else
    cfunc = dlsym(plugin,"eventFunction");
#endif
    // std::cout << "plugin:   " << cfunc << std::endl;
    if (!cfunc)
    {
		std::cout << "Can't find plugin function in plugin "
			<< '"' << path << "\"" << std::endl
#ifdef _WIN32
            << GetLastErrorAsString()
#else
            << dlerror()
#endif
			<< std::endl;
		plugin = 0;
		return;

    }
    eventFunction = (eventfunc_t)cfunc;
    
#ifdef _WIN32
	cfunc = GetProcAddress((HMODULE)plugin, "spikeFunction");
#else
    cfunc = dlsym(plugin,"spikeFunction");
#endif
    // std::cout << "plugin:   " << cfunc << std::endl;
    if (!cfunc)
    {
		std::cout << "Can't find plugin function in plugin "
			<< '"' << path << "\"" << std::endl
#ifdef _WIN32
            << GetLastErrorAsString()
#else
            << dlerror()
#endif
			<< std::endl;
		plugin = 0;
		return;
    }
    spikeFunction = (spikefunc_t)cfunc;
    
    // CJB added end

#ifdef _WIN32
	cfunc = GetProcAddress((HMODULE)plugin, "setIntParam");
#else
	cfunc = dlsym(plugin, "setIntParam");
#endif
    // std::cout << "plugin:   " << cfunc << std::endl;
    if (!cfunc)
    {
    	std::cout << "Can't find setIntParam function in plugin "
        << '"' << path << "\"" << std::endl
#ifdef _WIN32
        << GetLastErrorAsString()
#else
        << dlerror()
#endif
        << std::endl;
    	plugin = 0;
    	return;
    }
    setIntParamFunction = (setintparamfunc_t)cfunc;
    
#ifdef _WIN32
	cfunc = GetProcAddress((HMODULE)plugin, "setFloatParam");
#else
	cfunc = dlsym(plugin, "setFloatParam");
#endif
    // std::cout << "plugin:   " << cfunc << std::endl;
    if (!cfunc)
    {
		std::cout << "Can't find setFloatParam function in plugin "
			<< '"' << path << "\"" << std::endl
#ifdef _WIN32
            << GetLastErrorAsString()
#else
            << dlerror()
#endif
			<< std::endl;
		plugin = 0;
		return;
    }

    setFloatParamFunction = (setfloatparamfunc_t)cfunc;

#ifdef _WIN32
	cfunc = GetProcAddress((HMODULE)plugin, "getIntParam");
#else
	cfunc = dlsym(plugin, "getIntParam");
#endif

    // std::cout << "plugin:   " << cfunc << std::endl;
    if (!cfunc)
    {
        std::cout << "Can't find getIntParam function in plugin "
        << '"' << path << "\"" << std::endl
#ifdef _WIN32
        << GetLastErrorAsString()
#else
        << dlerror()
#endif
        << std::endl;
        plugin = 0;
        return;
    }
    getIntParamFunction = (getintparamfunc_t)cfunc;
    
#ifdef _WIN32
	cfunc = GetProcAddress((HMODULE)plugin, "getFloatParam");
#else
	cfunc = dlsym(plugin, "getFloatParam");
#endif
    // std::cout << "plugin:   " << cfunc << std::endl;
    if (!cfunc)
    {
		std::cout << "Can't find getFloatParam function in plugin "
			<< '"' << path << "\"" << std::endl
#ifdef _WIN32
            << GetLastErrorAsString()
#else
            << dlerror()
#endif
			<< std::endl;
		plugin = 0;
		return;
    }
    
    getFloatParamFunction = (getfloatparamfunc_t)cfunc;

    
// now the API should be fully loaded
    
    PyEval_RestoreThread(GUIThreadState);
    // initialize the plugin
#ifdef PYTHON_DEBUG
    std::cout << "before initplugin" << std::endl; // DEBUG
#endif 
    
    (*initF)();

#ifdef PYTHON_DEBUG
    std::cout << "after initplugin" << std::endl; // DEBUG
#endif
    
    (*pluginStartupFunction)(getSampleRate());
    
    // load the parameter configuration
    numPythonParams = (*getParamNumFunction)();
#ifdef PYTHON_DEBUG
    std::cout << "the plugin wants " << numPythonParams
        << " parameters" << std::endl;
#endif
    params = (ParamConfig *)calloc(numPythonParams, sizeof(ParamConfig));
    paramsControl = (Component **)calloc(numPythonParams, sizeof(Component *));
    
    (*getParamConfigFunction)(params);
#ifdef PYTHON_DEBUG
    std::cout << "release paramconfig" << std::endl;
#endif
    
    for(int i = 0; i < numPythonParams; i++)
    {
#ifdef PYTHON_DEBUG
        std::cout << "param " << i << " is a " << params[i].type << std::endl;
        std::cout << "it is named: " << params[i].name << std::endl << std::endl;
#endif
        switch (params[i].type) {
            case TOGGLE:
                paramsControl[i] = dynamic_cast<PythonEditor *>(getEditor())->addToggleButton(String(params[i].name), params[i].isEnabled);
                break;
            case INT_SET:
                paramsControl[i] = dynamic_cast<PythonEditor *>(getEditor())->addComboBox(String(params[i].name), params[i].nEntries, params[i].entries);
                break;
            case FLOAT_RANGE:
                paramsControl[i] = dynamic_cast<PythonEditor *>(getEditor())->addSlider(String(params[i].name), params[i].rangeMin, params[i].rangeMax, params[i].startValue);
                break;
            default:
                break;
        }
    }
    GUIThreadState = PyEval_SaveThread();
}


String PythonPlugin::getFile()
{
    return filePath;
}

void PythonPlugin::updateSettings()
{

}

void PythonPlugin::setIntPythonParameter(String name, int value)
{
    
#ifdef _WIN32
#else
#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    std::cout << "in setintparam pthread_threadid_np()=" << tid << std::endl;
#endif
#endif
    
    PyEval_RestoreThread(GUIThreadState);
    (*setIntParamFunction)(name.getCharPointer().getAddress(), value);
    GUIThreadState = PyEval_SaveThread();
}

void PythonPlugin::setFloatPythonParameter(String name, float value)
{

#ifdef _WIN32
#else
#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    std::cout << "in setfloatparam pthread_threadid_np()=" << tid << std::endl;
#endif
#endif
    PyEval_RestoreThread(GUIThreadState);
    (*setFloatParamFunction)(name.getCharPointer().getAddress(), value);
    GUIThreadState = PyEval_SaveThread();
}

int PythonPlugin::getIntPythonParameter(String name)
{
#ifdef _WIN32
#else
#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    std::cout << "in getintparam pthread_threadid_np()=" << tid << std::endl;
#endif
#endif

    int value;
    PyEval_RestoreThread(GUIThreadState);
    value = (*getIntParamFunction)(name.getCharPointer().getAddress());
    GUIThreadState = PyEval_SaveThread();
    return value;
}

float PythonPlugin::getFloatPythonParameter(String name)
{
    
#ifdef _WIN32
#else
#ifdef PYTHON_DEBUG
#if defined(__linux__)
    pid_t tid;
    tid = syscall(SYS_gettid);
#else
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
#endif
    std::cout << "in getfloatparam pthread_threadid_np()=" << tid << std::endl;
#endif
#endif
    
    PyEval_RestoreThread(GUIThreadState);
    float value;
    value = (*getFloatParamFunction)(name.getCharPointer().getAddress());
    GUIThreadState = PyEval_SaveThread();
    return value;
}

//saving settings


void PythonPlugin::saveCustomParametersToXml (XmlElement* parentElement)
{
    XmlElement* mainNode = parentElement->createNewChildElement ("PYTHONPLUGIN");
    mainNode->setAttribute ("filepath", filePath);
}

void PythonPlugin::loadCustomParametersFromXml()
{
    if (parametersAsXml)
    {
        //PythonEditor* ed = (PythonEditor*) getEditor();
        
        forEachXmlChildElement(*parametersAsXml, mainNode)
        {
            if (mainNode->hasTagName("PYTHONPLUGIN"))
            {
                filePath = mainNode->getStringAttribute("filepath");
                std::cout<<"set file path to: " << filePath << "\n";
               
            }
        }
    }
}



