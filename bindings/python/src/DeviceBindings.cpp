#include "DeviceBindings.hpp"

// depthai
#include "depthai/device/CrashDump.hpp"
#include "depthai/device/Device.hpp"
#include "depthai/device/EepromError.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/utility/Clock.hpp"
#include "depthai/utility/CompilerWarnings.hpp"
#include "depthai/xlink/XLinkConnection.hpp"

// std::chrono bindings
#include <pybind11/chrono.h>
// py::detail
#include <pybind11/detail/common.h>
// hedley
#include <hedley/hedley.h>
// STL Bind
#include <pybind11/pytypes.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::unordered_map<std::int8_t, dai::BoardConfig::GPIO>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::int8_t, dai::BoardConfig::UART>);

// Searches for available devices (as Device constructor)
// but pooling, to check for python interrupts, and releases GIL in between

template <typename DEVICE, class... Args>
static auto deviceSearchHelper(Args&&... args) {
    bool found;
    dai::DeviceInfo deviceInfo;
    // releases python GIL
    py::gil_scoped_release release;
    std::tie(found, deviceInfo) = DEVICE::getAnyAvailableDevice(DEVICE::getDefaultSearchTime(), []() {
        py::gil_scoped_acquire acquire;
        if(PyErr_CheckSignals() != 0) throw py::error_already_set();
    });

    // if no devices found, then throw
    if(!found) {
        auto numConnected = DEVICE::getAllAvailableDevices().size();
        if(numConnected > 0) {
            throw std::runtime_error("No available devices (" + std::to_string(numConnected) + " connected, but in use)");
        } else {
            throw std::runtime_error("No available devices");
        }
    }

    return deviceInfo;
}

// static std::vector<std::string> deviceGetQueueEventsHelper(dai::Device& d, const std::vector<std::string>& queueNames, std::size_t maxNumEvents,
// std::chrono::microseconds timeout){
//     using namespace std::chrono;

//     // if timeout < 0, unlimited timeout
//     bool unlimitedTimeout = timeout < microseconds(0);
//     auto startTime = steady_clock::now();
//     do {
//         {
//             // releases python GIL
//             py::gil_scoped_release release;
//             // block for 100ms
//             auto events = d.getQueueEvents(queueNames, maxNumEvents, std::chrono::milliseconds(100));
//             if(!events.empty()) return events;
//         }
//         // reacquires python GIL for PyErr_CheckSignals call
//         // check if interrupt triggered in between
//         if (PyErr_CheckSignals() != 0) throw py::error_already_set();
//     } while(unlimitedTimeout || steady_clock::now() - startTime < timeout);

//     return std::vector<std::string>();
// }

template <typename D, typename ARG>
static void bindConstructors(ARG& arg) {
    using namespace dai;

    arg.def(py::init([]() {
                auto dev = deviceSearchHelper<D>();
                py::gil_scoped_release release;
                return std::make_unique<D>(dev);
            }),
            DOC(dai, DeviceBase, DeviceBase))
        .def(py::init([](UsbSpeed maxUsbSpeed) {
                 auto dev = deviceSearchHelper<D>();
                 py::gil_scoped_release release;
                 return std::make_unique<D>(dev, maxUsbSpeed);
             }),
             py::arg("maxUsbSpeed"),
             DOC(dai, DeviceBase, DeviceBase, 2))
        .def(py::init([](const DeviceInfo& deviceInfo, UsbSpeed maxUsbSpeed) {
                 py::gil_scoped_release release;
                 return std::make_unique<D>(deviceInfo, maxUsbSpeed);
             }),
             py::arg("deviceInfo"),
             py::arg("maxUsbSpeed"),
             DOC(dai, DeviceBase, DeviceBase, 3))
        .def(py::init([](const DeviceInfo& deviceInfo, std::filesystem::path pathToCmd) {
                 py::gil_scoped_release release;
                 return std::make_unique<D>(deviceInfo, pathToCmd);
             }),
             py::arg("deviceDesc"),
             py::arg("pathToCmd"),
             DOC(dai, DeviceBase, DeviceBase, 4))
        .def(py::init([](typename D::Config config) {
                 auto dev = deviceSearchHelper<D>();
                 py::gil_scoped_release release;
                 return std::make_unique<D>(config, dev);
             }),
             py::arg("config"),
             DOC(dai, DeviceBase, DeviceBase, 5))
        .def(py::init([](typename D::Config config, const DeviceInfo& deviceInfo) {
                 py::gil_scoped_release release;
                 return std::make_unique<D>(config, deviceInfo);
             }),
             py::arg("config"),
             py::arg("deviceInfo"),
             DOC(dai, DeviceBase, DeviceBase, 6))

        // DeviceInfo version
        .def(py::init([](const DeviceInfo& deviceInfo) {
                 py::gil_scoped_release release;
                 return std::make_unique<D>(deviceInfo);
             }),
             py::arg("deviceInfo"),
             DOC(dai, DeviceBase, DeviceBase, 7))
        .def(py::init([](const DeviceInfo& deviceInfo, UsbSpeed maxUsbSpeed) {
                 py::gil_scoped_release release;
                 return std::make_unique<D>(deviceInfo, maxUsbSpeed);
             }),
             py::arg("deviceInfo"),
             py::arg("maxUsbSpeed"),
             DOC(dai, DeviceBase, DeviceBase, 8))

        // name or device id version
        .def(py::init([](std::string nameOrDeviceId) {
                 py::gil_scoped_release release;
                 return std::make_unique<D>(std::move(nameOrDeviceId));
             }),
             py::arg("nameOrDeviceId"),
             DOC(dai, DeviceBase, DeviceBase, 9))
        .def(py::init([](std::string nameOrDeviceId, UsbSpeed maxUsbSpeed) {
                 py::gil_scoped_release release;
                 return std::make_unique<D>(std::move(nameOrDeviceId), maxUsbSpeed);
             }),
             py::arg("nameOrDeviceId"),
             py::arg("maxUsbSpeed"),
             DOC(dai, DeviceBase, DeviceBase, 10));
}

void DeviceBindings::bind(pybind11::module& m, void* pCallstack) {
    using namespace dai;

    // Type definitions
    py::class_<DeviceBase, std::shared_ptr<DeviceBase>> deviceBase(m, "DeviceBase", DOC(dai, DeviceBase));
    py::class_<Device, DeviceBase, std::shared_ptr<Device>> device(m, "Device", DOC(dai, Device));
    py::class_<Device::Config> deviceConfig(device, "Config", DOC(dai, DeviceBase, Config));
    py::class_<CrashDump> crashDump(m, "CrashDump", DOC(dai, CrashDump));
    py::class_<CrashDump::CrashReport> crashReport(crashDump, "CrashReport", DOC(dai, CrashDump, CrashReport));
    py::class_<CrashDump::CrashReport::ErrorSourceInfo> errorSourceInfo(crashReport, "ErrorSourceInfo", DOC(dai, CrashDump, CrashReport, ErrorSourceInfo));
    py::class_<CrashDump::CrashReport::ErrorSourceInfo::AssertContext> assertContext(
        errorSourceInfo, "AssertContext", DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, AssertContext));
    py::class_<CrashDump::CrashReport::ErrorSourceInfo::TrapContext> trapContext(
        errorSourceInfo, "TrapContext", DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, TrapContext));
    py::class_<CrashDump::CrashReport::ThreadCallstack> threadCallstack(crashReport, "ThreadCallstack", DOC(dai, CrashDump, CrashReport, ThreadCallstack));
    py::class_<CrashDump::CrashReport::ThreadCallstack::CallstackContext> callstackContext(
        threadCallstack, "CallstackContext", DOC(dai, CrashDump, CrashReport, ThreadCallstack, CallstackContext));
    py::class_<BoardConfig> boardConfig(m, "BoardConfig", DOC(dai, BoardConfig));
    py::class_<BoardConfig::USB> boardConfigUsb(boardConfig, "USB", DOC(dai, BoardConfig, USB));
    py::class_<BoardConfig::Network> boardConfigNetwork(boardConfig, "Network", DOC(dai, BoardConfig, Network));
    py::class_<BoardConfig::GPIO> boardConfigGpio(boardConfig, "GPIO", DOC(dai, BoardConfig, GPIO));
    py::enum_<dai::Device::ReconnectionStatus> enumReconnectionStatus(device, "ReconnectionStatus", DOC(dai, DeviceBase, ReconnectionStatus));
    py::enum_<BoardConfig::GPIO::Mode> boardConfigGpioMode(boardConfigGpio, "Mode", DOC(dai, BoardConfig, GPIO, Mode));
    py::enum_<BoardConfig::GPIO::Direction> boardConfigGpioDirection(boardConfigGpio, "Direction", DOC(dai, BoardConfig, GPIO, Direction));
    py::enum_<BoardConfig::GPIO::Level> boardConfigGpioLevel(boardConfigGpio, "Level", DOC(dai, BoardConfig, GPIO, Level));
    py::enum_<BoardConfig::GPIO::Pull> boardConfigGpioPull(boardConfigGpio, "Pull", DOC(dai, BoardConfig, GPIO, Pull));
    py::enum_<BoardConfig::GPIO::Drive> boardConfigGpioDrive(boardConfigGpio, "Drive", DOC(dai, BoardConfig, GPIO, Drive));
    py::class_<BoardConfig::UART> boardConfigUart(boardConfig, "UART", DOC(dai, BoardConfig, UART));
    py::class_<BoardConfig::UVC> boardConfigUvc(boardConfig, "UVC", DOC(dai, BoardConfig, UVC));
    py::enum_<Platform> platform(m, "Platform", DOC(dai, Platform));
    struct PyClock {};
    py::class_<PyClock> clock(m, "Clock");

    py::bind_map<std::unordered_map<std::int8_t, dai::BoardConfig::GPIO>>(boardConfig, "GPIOMap");
    py::bind_map<std::unordered_map<std::int8_t, dai::BoardConfig::UART>>(boardConfig, "UARTMap");

    // pybind11 limitation of having actual classes as exceptions
    // Possible but requires a larger workaround
    // https://stackoverflow.com/questions/62087383/how-can-you-bind-exceptions-with-custom-fields-and-constructors-in-pybind11-and

    // Bind EepromError
    auto eepromError = py::register_exception<EepromError>(m, "EepromError", PyExc_RuntimeError);

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    // Call the rest of the type defines, then perform the actual bindings
    Callstack* callstack = (Callstack*)pCallstack;
    auto cb = callstack->top();
    callstack->pop();
    cb(m, pCallstack);
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    // Bind BoardConfig::USB
    boardConfigUsb.def(py::init<>())
        .def_readwrite("vid", &BoardConfig::USB::vid)
        .def_readwrite("pid", &BoardConfig::USB::pid)
        .def_readwrite("flashBootedVid", &BoardConfig::USB::flashBootedVid)
        .def_readwrite("flashBootedPid", &BoardConfig::USB::flashBootedPid)
        .def_readwrite("maxSpeed", &BoardConfig::USB::maxSpeed)
        .def_readwrite("productName", &BoardConfig::USB::productName)
        .def_readwrite("manufacturer", &BoardConfig::USB::manufacturer);

    // Bind BoardConfig::Network
    boardConfigNetwork.def(py::init<>())
        .def_readwrite("mtu", &BoardConfig::Network::mtu)
        .def_readwrite("xlinkTcpNoDelay", &BoardConfig::Network::xlinkTcpNoDelay);

    // GPIO Mode
    boardConfigGpioMode.value("ALT_MODE_0", BoardConfig::GPIO::ALT_MODE_0, DOC(dai, BoardConfig, GPIO, Mode, ALT_MODE_0))
        .value("ALT_MODE_1", BoardConfig::GPIO::ALT_MODE_1, DOC(dai, BoardConfig, GPIO, Mode, ALT_MODE_1))
        .value("ALT_MODE_2", BoardConfig::GPIO::ALT_MODE_2, DOC(dai, BoardConfig, GPIO, Mode, ALT_MODE_2))
        .value("ALT_MODE_3", BoardConfig::GPIO::ALT_MODE_3, DOC(dai, BoardConfig, GPIO, Mode, ALT_MODE_3))
        .value("ALT_MODE_4", BoardConfig::GPIO::ALT_MODE_4, DOC(dai, BoardConfig, GPIO, Mode, ALT_MODE_4))
        .value("ALT_MODE_5", BoardConfig::GPIO::ALT_MODE_5, DOC(dai, BoardConfig, GPIO, Mode, ALT_MODE_5))
        .value("ALT_MODE_6", BoardConfig::GPIO::ALT_MODE_6, DOC(dai, BoardConfig, GPIO, Mode, ALT_MODE_6))
        .value("DIRECT", BoardConfig::GPIO::DIRECT, DOC(dai, BoardConfig, GPIO, Mode, DIRECT))
        .export_values();

    // GPIO Direction
    boardConfigGpioDirection.value("INPUT", BoardConfig::GPIO::INPUT, DOC(dai, BoardConfig, GPIO, Direction, INPUT))
        .value("OUTPUT", BoardConfig::GPIO::OUTPUT, DOC(dai, BoardConfig, GPIO, Direction, OUTPUT))
        .export_values();

    // GPIO Level
    boardConfigGpioLevel.value("LOW", BoardConfig::GPIO::LOW, DOC(dai, BoardConfig, GPIO, Level, LOW))
        .value("HIGH", BoardConfig::GPIO::HIGH, DOC(dai, BoardConfig, GPIO, Level, HIGH))
        .export_values();

    // GPIO Pull
    boardConfigGpioPull.value("NO_PULL", BoardConfig::GPIO::NO_PULL, DOC(dai, BoardConfig, GPIO, Pull, NO_PULL))
        .value("PULL_UP", BoardConfig::GPIO::PULL_UP, DOC(dai, BoardConfig, GPIO, Pull, PULL_UP))
        .value("PULL_DOWN", BoardConfig::GPIO::PULL_DOWN, DOC(dai, BoardConfig, GPIO, Pull, PULL_DOWN))
        .value("BUS_KEEPER", BoardConfig::GPIO::BUS_KEEPER, DOC(dai, BoardConfig, GPIO, Pull, BUS_KEEPER))
        .export_values();

    // GPIO Drive
    boardConfigGpioDrive.value("MA_2", BoardConfig::GPIO::MA_2, DOC(dai, BoardConfig, GPIO, Drive, MA_2))
        .value("MA_4", BoardConfig::GPIO::MA_4, DOC(dai, BoardConfig, GPIO, Drive, MA_4))
        .value("MA_8", BoardConfig::GPIO::MA_8, DOC(dai, BoardConfig, GPIO, Drive, MA_8))
        .value("MA_12", BoardConfig::GPIO::MA_12, DOC(dai, BoardConfig, GPIO, Drive, MA_12))
        .export_values();

    // Bind BoardConfig::GPIO
    boardConfigGpio.def(py::init<>())
        .def(py::init<BoardConfig::GPIO::Direction>())
        .def(py::init<BoardConfig::GPIO::Direction, BoardConfig::GPIO::Level>())
        .def(py::init<BoardConfig::GPIO::Direction, BoardConfig::GPIO::Level, BoardConfig::GPIO::Pull>())
        .def(py::init<BoardConfig::GPIO::Direction, BoardConfig::GPIO::Mode>())
        .def(py::init<BoardConfig::GPIO::Direction, BoardConfig::GPIO::Mode, BoardConfig::GPIO::Pull>())
        .def_readwrite("mode", &BoardConfig::GPIO::mode)
        .def_readwrite("direction", &BoardConfig::GPIO::direction)
        .def_readwrite("level", &BoardConfig::GPIO::level)
        .def_readwrite("pull", &BoardConfig::GPIO::pull)
        .def_readwrite("drive", &BoardConfig::GPIO::drive)
        .def_readwrite("schmitt", &BoardConfig::GPIO::schmitt)
        .def_readwrite("slewFast", &BoardConfig::GPIO::slewFast);

    // Bind BoardConfig::UART
    boardConfigUart.def(py::init<>()).def_readwrite("tmp", &BoardConfig::UART::tmp);

    // Bind BoardConfig::UVC
    boardConfigUvc.def(py::init<>())
        .def(py::init<uint16_t, uint16_t>())
        .def_readwrite("cameraName", &BoardConfig::UVC::cameraName)
        .def_readwrite("width", &BoardConfig::UVC::width)
        .def_readwrite("height", &BoardConfig::UVC::height)
        .def_readwrite("frameType", &BoardConfig::UVC::frameType)
        .def_readwrite("enable", &BoardConfig::UVC::enable);

    // Bind Platform
    platform.value("RVC2", Platform::RVC2, DOC(dai, Platform, RVC2))
        .value("RVC3", Platform::RVC3, DOC(dai, Platform, RVC3))
        .value("RVC4", Platform::RVC4, DOC(dai, Platform, RVC4));

    // Bind Device::ReconnectionStatus
    enumReconnectionStatus.value("RECONNECT_FAILED", Device::ReconnectionStatus::RECONNECT_FAILED);
    enumReconnectionStatus.value("RECONNECTED", Device::ReconnectionStatus::RECONNECTED);
    enumReconnectionStatus.value("RECONNECTING", Device::ReconnectionStatus::RECONNECTING);

    // Platform - string conversion
    m.def("platform2string", &platform2string, DOC(dai, platform2string));
    m.def("string2platform", &string2platform, DOC(dai, string2platform));

    // Bind BoardConfig
    boardConfig.def(py::init<>())
        .def_readwrite("usb", &BoardConfig::usb, DOC(dai, BoardConfig, usb))
        .def_readwrite("network", &BoardConfig::network, DOC(dai, BoardConfig, network))
        .def_readwrite("sysctl", &BoardConfig::sysctl, DOC(dai, BoardConfig, sysctl))
        .def_readwrite("watchdogTimeoutMs", &BoardConfig::watchdogTimeoutMs, DOC(dai, BoardConfig, watchdogTimeoutMs))
        .def_readwrite("watchdogInitialDelayMs", &BoardConfig::watchdogInitialDelayMs, DOC(dai, BoardConfig, watchdogInitialDelayMs))
        .def_readwrite("gpio", &BoardConfig::gpio, DOC(dai, BoardConfig, gpio))
        .def_readwrite("uart", &BoardConfig::uart, DOC(dai, BoardConfig, uart))
        .def_readwrite("pcieInternalClock", &BoardConfig::pcieInternalClock, DOC(dai, BoardConfig, pcieInternalClock))
        .def_readwrite("usb3PhyInternalClock", &BoardConfig::usb3PhyInternalClock, DOC(dai, BoardConfig, usb3PhyInternalClock))
        .def_readwrite("mipi4LaneRgb", &BoardConfig::mipi4LaneRgb, DOC(dai, BoardConfig, mipi4LaneRgb))
        .def_readwrite("emmc", &BoardConfig::emmc, DOC(dai, BoardConfig, emmc))
        .def_readwrite("logPath", &BoardConfig::logPath, DOC(dai, BoardConfig, logPath))
        .def_readwrite("logSizeMax", &BoardConfig::logSizeMax, DOC(dai, BoardConfig, logSizeMax))
        .def_readwrite("logVerbosity", &BoardConfig::logVerbosity, DOC(dai, BoardConfig, logVerbosity))
        .def_readwrite("logDevicePrints", &BoardConfig::logDevicePrints, DOC(dai, BoardConfig, logDevicePrints))
        .def_readwrite("uvc", &BoardConfig::uvc, DOC(dai, BoardConfig, uvc));

    // Bind Device::Config
    deviceConfig.def(py::init<>())
        .def_readwrite("version", &Device::Config::version)
        .def_readwrite("board", &Device::Config::board)
        .def_readwrite("nonExclusiveMode", &Device::Config::nonExclusiveMode)
        .def_readwrite("outputLogLevel", &Device::Config::outputLogLevel)
        .def_readwrite("logLevel", &Device::Config::logLevel);

    // Bind CrashDump
    crashDump.def(py::init<>())
        .def("serializeToJson", &CrashDump::serializeToJson, DOC(dai, CrashDump, serializeToJson))

        .def_readwrite("crashReports", &CrashDump::crashReports, DOC(dai, CrashDump, crashReports))
        .def_readwrite("depthaiCommitHash", &CrashDump::depthaiCommitHash, DOC(dai, CrashDump, depthaiCommitHash))
        .def_readwrite("deviceId", &CrashDump::deviceId, DOC(dai, CrashDump, deviceId));

    crashReport.def(py::init<>())
        .def_readwrite("processor", &CrashDump::CrashReport::processor, DOC(dai, CrashDump, CrashReport, processor))
        .def_readwrite("errorSource", &CrashDump::CrashReport::errorSource, DOC(dai, CrashDump, CrashReport, errorSource))
        .def_readwrite("crashedThreadId", &CrashDump::CrashReport::crashedThreadId, DOC(dai, CrashDump, CrashReport, crashedThreadId))
        .def_readwrite("threadCallstack", &CrashDump::CrashReport::threadCallstack, DOC(dai, CrashDump, CrashReport, threadCallstack));

    errorSourceInfo.def(py::init<>())
        .def_readwrite(
            "assertContext", &CrashDump::CrashReport::ErrorSourceInfo::assertContext, DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, assertContext))
        .def_readwrite("trapContext", &CrashDump::CrashReport::ErrorSourceInfo::trapContext, DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, trapContext))
        .def_readwrite("errorId", &CrashDump::CrashReport::ErrorSourceInfo::errorId, DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, errorId));

    assertContext.def(py::init<>())
        .def_readwrite("fileName",
                       &CrashDump::CrashReport::ErrorSourceInfo::AssertContext::fileName,
                       DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, AssertContext, fileName))
        .def_readwrite("functionName",
                       &CrashDump::CrashReport::ErrorSourceInfo::AssertContext::functionName,
                       DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, AssertContext, functionName))
        .def_readwrite(
            "line", &CrashDump::CrashReport::ErrorSourceInfo::AssertContext::line, DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, AssertContext, line));

    trapContext.def(py::init<>())
        .def_readwrite("trapNumber",
                       &CrashDump::CrashReport::ErrorSourceInfo::TrapContext::trapNumber,
                       DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, TrapContext, trapNumber))
        .def_readwrite("trapAddress",
                       &CrashDump::CrashReport::ErrorSourceInfo::TrapContext::trapAddress,
                       DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, TrapContext, trapAddress))
        .def_readwrite("trapName",
                       &CrashDump::CrashReport::ErrorSourceInfo::TrapContext::trapName,
                       DOC(dai, CrashDump, CrashReport, ErrorSourceInfo, TrapContext, trapName));

    threadCallstack.def(py::init<>())
        .def_readwrite("threadId", &CrashDump::CrashReport::ThreadCallstack::threadId, DOC(dai, CrashDump, CrashReport, ThreadCallstack, threadId))
        .def_readwrite("threadName", &CrashDump::CrashReport::ThreadCallstack::threadName, DOC(dai, CrashDump, CrashReport, ThreadCallstack, threadName))
        .def_readwrite("stackBottom", &CrashDump::CrashReport::ThreadCallstack::stackBottom, DOC(dai, CrashDump, CrashReport, ThreadCallstack, stackBottom))
        .def_readwrite("stackTop", &CrashDump::CrashReport::ThreadCallstack::stackTop, DOC(dai, CrashDump, CrashReport, ThreadCallstack, stackTop))
        .def_readwrite("stackPointer", &CrashDump::CrashReport::ThreadCallstack::stackPointer, DOC(dai, CrashDump, CrashReport, ThreadCallstack, stackPointer))
        .def_readwrite("instructionPointer",
                       &CrashDump::CrashReport::ThreadCallstack::instructionPointer,
                       DOC(dai, CrashDump, CrashReport, ThreadCallstack, instructionPointer))
        .def_readwrite("threadStatus", &CrashDump::CrashReport::ThreadCallstack::threadStatus, DOC(dai, CrashDump, CrashReport, ThreadCallstack, threadStatus))
        .def_readwrite("callStack", &CrashDump::CrashReport::ThreadCallstack::callStack, DOC(dai, CrashDump, CrashReport, ThreadCallstack, callStack));

    callstackContext.def(py::init<>())
        .def_readwrite("callSite",
                       &CrashDump::CrashReport::ThreadCallstack::CallstackContext::callSite,
                       DOC(dai, CrashDump, CrashReport, ThreadCallstack, CallstackContext, callSite))
        .def_readwrite("calledTarget",
                       &CrashDump::CrashReport::ThreadCallstack::CallstackContext::calledTarget,
                       DOC(dai, CrashDump, CrashReport, ThreadCallstack, CallstackContext, calledTarget))
        .def_readwrite("framePointer",
                       &CrashDump::CrashReport::ThreadCallstack::CallstackContext::framePointer,
                       DOC(dai, CrashDump, CrashReport, ThreadCallstack, CallstackContext, framePointer))
        .def_readwrite("context",
                       &CrashDump::CrashReport::ThreadCallstack::CallstackContext::context,
                       DOC(dai, CrashDump, CrashReport, ThreadCallstack, CallstackContext, context));

    // Bind constructors
    bindConstructors<DeviceBase>(deviceBase);
    // Bind the rest
    deviceBase
        // Python only methods
        .def("__enter__", [](DeviceBase& d) -> DeviceBase& { return d; })
        .def("__exit__",
             [](DeviceBase& d, py::object type, py::object value, py::object traceback) {
                 py::gil_scoped_release release;
                 d.close();
             })
        .def(
            "close",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                d.close();
            },
            "Closes the connection to device. Better alternative is the usage of context manager: `with depthai.Device(pipeline) as device:`")
        .def(
            "isClosed",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.isClosed();
            },
            DOC(dai, DeviceBase, isClosed))
        .def("setMaxReconnectionAttempts",
             &DeviceBase::setMaxReconnectionAttempts,
             py::arg("maxAttempts"),
             py::arg("callback") = py::none(),
             DOC(dai, DeviceBase, setMaxReconnectionAttempts))

        // dai::Device methods
        // static
        .def_static(
            "getAnyAvailableDevice",
            [](std::chrono::milliseconds ms) { return DeviceBase::getAnyAvailableDevice(ms); },
            py::arg("timeout"),
            DOC(dai, DeviceBase, getAnyAvailableDevice))
        .def_static(
            "getAnyAvailableDevice", []() { return DeviceBase::getAnyAvailableDevice(); }, DOC(dai, DeviceBase, getAnyAvailableDevice, 2))
        .def_static("getFirstAvailableDevice",
                    &DeviceBase::getFirstAvailableDevice,
                    py::arg("skipInvalidDevices") = true,
                    DOC(dai, DeviceBase, getFirstAvailableDevice))
        .def_static("getAllAvailableDevices", &DeviceBase::getAllAvailableDevices, DOC(dai, DeviceBase, getAllAvailableDevices))
        .def_static("getEmbeddedDeviceBinary",
                    py::overload_cast<bool, OpenVINO::Version>(&DeviceBase::getEmbeddedDeviceBinary),
                    py::arg("usb2Mode"),
                    py::arg("version") = OpenVINO::VERSION_UNIVERSAL,
                    DOC(dai, DeviceBase, getEmbeddedDeviceBinary))
        .def_static("getEmbeddedDeviceBinary",
                    py::overload_cast<DeviceBase::Config>(&DeviceBase::getEmbeddedDeviceBinary),
                    py::arg("config"),
                    DOC(dai, DeviceBase, getEmbeddedDeviceBinary, 2))
        .def_static("getDeviceById", &DeviceBase::getDeviceById, py::arg("deviceId"), DOC(dai, DeviceBase, getDeviceById))
        .def_static("getAllConnectedDevices", &DeviceBase::getAllConnectedDevices, DOC(dai, DeviceBase, getAllConnectedDevices))
        .def_static("getGlobalProfilingData", &DeviceBase::getGlobalProfilingData, DOC(dai, DeviceBase, getGlobalProfilingData))

        // methods
        .def("getBootloaderVersion", &DeviceBase::getBootloaderVersion, DOC(dai, DeviceBase, getBootloaderVersion))
        .def(
            "isPipelineRunning",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.isPipelineRunning();
            },
            DOC(dai, DeviceBase, isPipelineRunning))

        // Doesn't require GIL release (eg, don't do RPC or long blocking things in background)
        .def("setLogOutputLevel", &DeviceBase::setLogOutputLevel, py::arg("level"), DOC(dai, DeviceBase, setLogOutputLevel))
        .def("getLogOutputLevel", &DeviceBase::getLogOutputLevel, DOC(dai, DeviceBase, getLogOutputLevel))

        // Requires GIL release
        .def(
            "setLogLevel",
            [](DeviceBase& d, LogLevel l) {
                py::gil_scoped_release release;
                d.setLogLevel(l);
            },
            py::arg("level"),
            DOC(dai, DeviceBase, setLogLevel))
        .def(
            "getLogLevel",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getLogLevel();
            },
            DOC(dai, DeviceBase, getLogLevel))
        .def(
            "setSystemInformationLoggingRate",
            [](DeviceBase& d, float hz) {
                py::gil_scoped_release release;
                d.setSystemInformationLoggingRate(hz);
            },
            py::arg("rateHz"),
            DOC(dai, DeviceBase, setSystemInformationLoggingRate))
        .def(
            "getSystemInformationLoggingRate",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getSystemInformationLoggingRate();
            },
            DOC(dai, DeviceBase, getSystemInformationLoggingRate))
        .def(
            "getCrashDump",
            [](DeviceBase& d, bool clearCrashDump) {
                py::gil_scoped_release release;
                return d.getCrashDump(clearCrashDump);
            },
            py::arg("clearCrashDump") = true,
            DOC(dai, DeviceBase, getCrashDump))
        .def(
            "hasCrashDump",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.hasCrashDump();
            },
            DOC(dai, DeviceBase, hasCrashDump))
        .def(
            "getConnectedCameras",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getConnectedCameras();
            },
            DOC(dai, DeviceBase, getConnectedCameras))
        .def(
            "getConnectionInterfaces",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getConnectionInterfaces();
            },
            DOC(dai, DeviceBase, getConnectionInterfaces))
        .def(
            "getConnectedCameraFeatures",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getConnectedCameraFeatures();
            },
            DOC(dai, DeviceBase, getConnectedCameraFeatures))
        .def(
            "getCameraSensorNames",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getCameraSensorNames();
            },
            DOC(dai, DeviceBase, getCameraSensorNames))
        .def(
            "getStereoPairs",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getStereoPairs();
            },
            DOC(dai, DeviceBase, getStereoPairs))
        .def(
            "getAvailableStereoPairs",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getAvailableStereoPairs();
            },
            DOC(dai, DeviceBase, getAvailableStereoPairs))
        .def(
            "getConnectedIMU",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getConnectedIMU();
            },
            DOC(dai, DeviceBase, getConnectedIMU))
        .def(
            "getIMUFirmwareVersion",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getIMUFirmwareVersion();
            },
            DOC(dai, DeviceBase, getIMUFirmwareVersion))
        .def(
            "getEmbeddedIMUFirmwareVersion",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getEmbeddedIMUFirmwareVersion();
            },
            DOC(dai, DeviceBase, getEmbeddedIMUFirmwareVersion))
        .def(
            "startIMUFirmwareUpdate",
            [](DeviceBase& d, bool forceUpdate) {
                py::gil_scoped_release release;
                return d.startIMUFirmwareUpdate(forceUpdate);
            },
            py::arg("forceUpdate") = false,
            DOC(dai, DeviceBase, startIMUFirmwareUpdate))
        .def(
            "getIMUFirmwareUpdateStatus",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getIMUFirmwareUpdateStatus();
            },
            DOC(dai, DeviceBase, getIMUFirmwareUpdateStatus))
        .def(
            "getDdrMemoryUsage",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getDdrMemoryUsage();
            },
            DOC(dai, DeviceBase, getDdrMemoryUsage))
        .def(
            "getCmxMemoryUsage",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getCmxMemoryUsage();
            },
            DOC(dai, DeviceBase, getCmxMemoryUsage))
        .def(
            "getLeonCssHeapUsage",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getLeonCssHeapUsage();
            },
            DOC(dai, DeviceBase, getLeonCssHeapUsage))
        .def(
            "getLeonMssHeapUsage",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getLeonMssHeapUsage();
            },
            DOC(dai, DeviceBase, getLeonMssHeapUsage))
        .def(
            "getChipTemperature",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getChipTemperature();
            },
            DOC(dai, DeviceBase, getChipTemperature))
        .def(
            "getLeonCssCpuUsage",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getLeonCssCpuUsage();
            },
            DOC(dai, DeviceBase, getLeonCssCpuUsage))
        .def(
            "getLeonMssCpuUsage",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getLeonMssCpuUsage();
            },
            DOC(dai, DeviceBase, getLeonMssCpuUsage))
        .def(
            "addLogCallback",
            [](DeviceBase& d, std::function<void(LogMessage)> callback) {
                py::gil_scoped_release release;
                return d.addLogCallback(callback);
            },
            py::arg("callback"),
            DOC(dai, DeviceBase, addLogCallback))
        .def(
            "removeLogCallback",
            [](DeviceBase& d, int cbId) {
                py::gil_scoped_release release;
                return d.removeLogCallback(cbId);
            },
            py::arg("callbackId"),
            DOC(dai, DeviceBase, removeLogCallback))
        .def(
            "getUsbSpeed",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getUsbSpeed();
            },
            DOC(dai, DeviceBase, getUsbSpeed))
        .def(
            "getDeviceInfo",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getDeviceInfo();
            },
            DOC(dai, DeviceBase, getDeviceInfo))
        .def(
            "getMxId",
            [](DeviceBase& d) {
                PyErr_WarnEx(PyExc_DeprecationWarning, "Use getDeviceId() instead.", 1);
                py::gil_scoped_release release;
                DEPTHAI_BEGIN_SUPPRESS_DEPRECATION_WARNING
                return d.getMxId();
                DEPTHAI_END_SUPPRESS_DEPRECATION_WARNING
            },
            DOC(dai, DeviceBase, getMxId))
        .def(
            "getDeviceId",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getDeviceId();
            },
            DOC(dai, DeviceBase, getDeviceId))
        .def(
            "getProfilingData",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getProfilingData();
            },
            DOC(dai, DeviceBase, getProfilingData))
        .def(
            "readCalibration",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.readCalibration();
            },
            DOC(dai, DeviceBase, readCalibration))
        .def(
            "tryFlashCalibration",
            [](DeviceBase& d, CalibrationHandler calibrationDataHandler) {
                py::gil_scoped_release release;
                return d.tryFlashCalibration(calibrationDataHandler);
            },
            py::arg("calibrationDataHandler"),
            DOC(dai, DeviceBase, tryFlashCalibration))
        .def(
            "setXLinkChunkSize",
            [](DeviceBase& d, int s) {
                py::gil_scoped_release release;
                d.setXLinkChunkSize(s);
            },
            py::arg("sizeBytes"),
            DOC(dai, DeviceBase, setXLinkChunkSize))
        .def(
            "getXLinkChunkSize",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getXLinkChunkSize();
            },
            DOC(dai, DeviceBase, getXLinkChunkSize))

        .def(
            "setIrLaserDotProjectorIntensity",
            [](DeviceBase& d, float intensity, int mask) {
                py::gil_scoped_release release;
                return d.setIrLaserDotProjectorIntensity(intensity, mask);
            },
            py::arg("intensity"),
            py::arg("mask") = -1,
            DOC(dai, DeviceBase, setIrLaserDotProjectorIntensity))
        .def(
            "setIrFloodLightIntensity",
            [](DeviceBase& d, float intensity, int mask) {
                py::gil_scoped_release release;
                return d.setIrFloodLightIntensity(intensity, mask);
            },
            py::arg("intensity"),
            py::arg("mask") = -1,
            DOC(dai, DeviceBase, setIrFloodLightIntensity))
        .def(
            "getIrDrivers",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.getIrDrivers();
            },
            DOC(dai, DeviceBase, getIrDrivers))
        .def(
            "isEepromAvailable",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.isEepromAvailable();
            },
            DOC(dai, DeviceBase, isEepromAvailable))
        .def(
            "flashCalibration",
            [](DeviceBase& d, CalibrationHandler ch) {
                py::gil_scoped_release release;
                return d.flashCalibration(ch);
            },
            DOC(dai, DeviceBase, flashCalibration))
        .def(
            "readCalibration2",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.readCalibration2();
            },
            DOC(dai, DeviceBase, readCalibration2))
        .def(
            "readCalibrationOrDefault",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.readCalibrationOrDefault();
            },
            DOC(dai, DeviceBase, readCalibrationOrDefault))
        .def(
            "factoryResetCalibration",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.factoryResetCalibration();
            },
            DOC(dai, DeviceBase, factoryResetCalibration))
        .def(
            "flashFactoryCalibration",
            [](DeviceBase& d, CalibrationHandler ch) {
                py::gil_scoped_release release;
                return d.flashFactoryCalibration(ch);
            },
            DOC(dai, DeviceBase, flashFactoryCalibration))
        .def(
            "readFactoryCalibration",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.readFactoryCalibration();
            },
            DOC(dai, DeviceBase, readFactoryCalibration))
        .def(
            "readFactoryCalibrationOrDefault",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.readFactoryCalibrationOrDefault();
            },
            DOC(dai, DeviceBase, readFactoryCalibrationOrDefault))
        .def(
            "readCalibrationRaw",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.readCalibrationRaw();
            },
            DOC(dai, DeviceBase, readCalibrationRaw))
        .def(
            "readFactoryCalibrationRaw",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.readFactoryCalibrationRaw();
            },
            DOC(dai, DeviceBase, readFactoryCalibrationRaw))
        .def(
            "flashEepromClear",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                d.flashEepromClear();
            },
            DOC(dai, DeviceBase, flashEepromClear))
        .def(
            "flashFactoryEepromClear",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                d.flashFactoryEepromClear();
            },
            DOC(dai, DeviceBase, flashFactoryEepromClear))
        .def(
            "setTimesync",
            [](DeviceBase& d, std::chrono::milliseconds p, int s, bool r) {
                py::gil_scoped_release release;
                return d.setTimesync(p, s, r);
            },
            DOC(dai, DeviceBase, setTimesync))
        .def(
            "setTimesync",
            [](DeviceBase& d, bool e) {
                py::gil_scoped_release release;
                return d.setTimesync(e);
            },
            py::arg("enable"),
            DOC(dai, DeviceBase, setTimesync, 2))
        .def(
            "getDeviceName",
            [](DeviceBase& d) {
                std::string name;
                {
                    py::gil_scoped_release release;
                    name = d.getDeviceName();
                }
                return py::bytes(name).attr("decode")("utf-8", "replace");
            },
            DOC(dai, DeviceBase, getDeviceName))
        .def(
            "getProductName",
            [](DeviceBase& d) {
                std::string name;
                {
                    py::gil_scoped_release release;
                    name = d.getProductName();
                }
                return py::bytes(name).attr("decode")("utf-8", "replace");
            },
            DOC(dai, DeviceBase, getProductName))
        .def(
            "crashDevice",
            [](DeviceBase& d) {
                py::gil_scoped_release release;
                return d.crashDevice();
            },
            DOC(dai, DeviceBase, crashDevice));

    // Bind constructors
    bindConstructors<Device>(device);

    device.def("getPlatform", &Device::getPlatform, DOC(dai, Device, getPlatform))
        .def("getPlatformAsString", &Device::getPlatformAsString, DOC(dai, Device, getPlatformAsString));

    clock.def("now", &Clock::now);
}
