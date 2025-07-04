#include "CommonBindings.hpp"

// Libraries
#include "hedley/hedley.h"

// depthai/
#include "depthai/common/CameraBoardSocket.hpp"
#include "depthai/common/CameraFeatures.hpp"
#include "depthai/common/CameraImageOrientation.hpp"
#include "depthai/common/CameraSensorType.hpp"
#include "depthai/common/ChipTemperature.hpp"
#include "depthai/common/ChipTemperatureS3.hpp"
#include "depthai/common/Color.hpp"
#include "depthai/common/Colormap.hpp"
#include "depthai/common/ConnectionInterface.hpp"
#include "depthai/common/CpuUsage.hpp"
#include "depthai/common/DetectionNetworkType.hpp"
#include "depthai/common/DetectionParserOptions.hpp"
#include "depthai/common/EepromData.hpp"
#include "depthai/common/FrameEvent.hpp"
#include "depthai/common/Interpolation.hpp"
#include "depthai/common/MemoryInfo.hpp"
#include "depthai/common/Point2f.hpp"
#include "depthai/common/Point3d.hpp"
#include "depthai/common/Point3f.hpp"
#include "depthai/common/Point3fRGBA.hpp"
#include "depthai/common/ProcessorType.hpp"
#include "depthai/common/Quaterniond.hpp"
#include "depthai/common/Rect.hpp"
#include "depthai/common/RotatedRect.hpp"
#include "depthai/common/Size2f.hpp"
#include "depthai/common/StereoPair.hpp"
#include "depthai/common/Timestamp.hpp"
#include "depthai/common/UsbSpeed.hpp"

// depthai
#include "depthai/common/CameraExposureOffset.hpp"
#include "depthai/common/CameraFeatures.hpp"
#include "depthai/common/StereoPair.hpp"
#include "depthai/utility/ProfilingData.hpp"

void CommonBindings::bind(pybind11::module& m, void* pCallstack) {
    using namespace dai;

    py::class_<Timestamp> timestamp(m, "Timestamp", DOC(dai, Timestamp));
    py::class_<Point2f> point2f(m, "Point2f", DOC(dai, Point2f));
    py::class_<Point3f> point3f(m, "Point3f", DOC(dai, Point3f));
    py::class_<Point3fRGBA> point3fRGBA(m, "Point3fRGBA", DOC(dai, Point3fRGBA));
    py::class_<Point3d> point3d(m, "Point3d", DOC(dai, Point3d));
    py::class_<Quaterniond> quaterniond(m, "Quaterniond", DOC(dai, Quaterniond));
    py::class_<Size2f> size2f(m, "Size2f", DOC(dai, Size2f));
    py::enum_<CameraBoardSocket> cameraBoardSocket(m, "CameraBoardSocket", DOC(dai, CameraBoardSocket));
    py::enum_<ConnectionInterface> connectionInterface(m, "connectionInterface", DOC(dai, ConnectionInterface));
    py::enum_<CameraSensorType> cameraSensorType(m, "CameraSensorType", DOC(dai, CameraSensorType));
    py::enum_<CameraImageOrientation> cameraImageOrientation(m, "CameraImageOrientation", DOC(dai, CameraImageOrientation));
    py::class_<CameraSensorConfig> cameraSensorConfig(m, "CameraSensorConfig", DOC(dai, CameraSensorConfig));
    py::class_<CameraFeatures> cameraFeatures(m, "CameraFeatures", DOC(dai, CameraFeatures));
    py::class_<MemoryInfo> memoryInfo(m, "MemoryInfo", DOC(dai, MemoryInfo));
    py::class_<ChipTemperature> chipTemperature(m, "ChipTemperature", DOC(dai, ChipTemperature));
    py::class_<ChipTemperatureS3> chipTemperatureS3(m, "ChipTemperatureS3", DOC(dai, ChipTemperatureS3));
    py::class_<CpuUsage> cpuUsage(m, "CpuUsage", DOC(dai, CpuUsage));
    py::enum_<CameraModel> cameraModel(m, "CameraModel", DOC(dai, CameraModel));
    py::class_<StereoRectification> stereoRectification(m, "StereoRectification", DOC(dai, StereoRectification));
    py::class_<Extrinsics> extrinsics(m, "Extrinsics", DOC(dai, Extrinsics));
    py::class_<CameraInfo> cameraInfo(m, "CameraInfo", DOC(dai, CameraInfo));
    py::class_<EepromData> eepromData(m, "EepromData", DOC(dai, EepromData));
    py::enum_<UsbSpeed> usbSpeed(m, "UsbSpeed", DOC(dai, UsbSpeed));
    py::enum_<ProcessorType> processorType(m, "ProcessorType");
    py::enum_<DetectionNetworkType> detectionNetworkType(m, "DetectionNetworkType");
    py::enum_<SerializationType> serializationType(m, "SerializationType");
    py::class_<DetectionParserOptions> detectionParserOptions(m, "DetectionParserOptions", DOC(dai, DetectionParserOptions));
    py::class_<RotatedRect> rotatedRect(m, "RotatedRect", DOC(dai, RotatedRect));
    py::class_<Rect> rect(m, "Rect", DOC(dai, Rect));
    py::class_<StereoPair> stereoPair(m, "StereoPair", DOC(dai, StereoPair));
    py::enum_<CameraExposureOffset> cameraExposureOffset(m, "CameraExposureOffset");
    py::class_<Color> color(m, "Color", DOC(dai, Color));
    py::enum_<Colormap> colormap(m, "Colormap", DOC(dai, Colormap));
    py::enum_<FrameEvent> frameEvent(m, "FrameEvent", DOC(dai, FrameEvent));
    py::class_<ProfilingData> profilingData(m, "ProfilingData", DOC(dai, ProfilingData));
    py::enum_<Interpolation> interpolation(m, "Interpolation", DOC(dai, Interpolation));

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    // Call the rest of the type defines, then perform the actual bindings
    Callstack* callstack = (Callstack*)pCallstack;
    auto cb = callstack->top();
    callstack->pop();
    cb(m, pCallstack);
    // Actual bindings
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////

    rotatedRect.def(py::init<>())
        .def(py::init<Point2f, Size2f, float>())
        .def(py::init<Rect, float>())
        .def_readwrite("center", &RotatedRect::center)
        .def_readwrite("size", &RotatedRect::size)
        .def_readwrite("angle", &RotatedRect::angle)
        .def("isNormalized", &RotatedRect::isNormalized, DOC(dai, RotatedRect, isNormalized))
        .def("normalize", &RotatedRect::normalize, py::arg("width"), py::arg("height"), DOC(dai, RotatedRect, normalize))
        .def("denormalize", &RotatedRect::denormalize, py::arg("width"), py::arg("height"), DOC(dai, RotatedRect, denormalize))
        .def("getPoints", &RotatedRect::getPoints, DOC(dai, RotatedRect, getPoints))
        .def("getOuterRect", &RotatedRect::getOuterRect, DOC(dai, RotatedRect, getOuterRect));

    rect.def(py::init<>())
        .def(py::init<float, float, float, float>())
        .def(py::init<float, float, float, float, bool>())
        .def(py::init<Point2f, Point2f>())
        .def(py::init<Point2f, Point2f, bool>())
        .def(py::init<Point2f, Size2f>())
        .def(py::init<Point2f, Size2f, bool>())

        .def("topLeft", &Rect::topLeft, DOC(dai, Rect, topLeft))
        .def("bottomRight", &Rect::bottomRight, DOC(dai, Rect, bottomRight))
        .def("size", &Rect::size, DOC(dai, Rect, size))
        .def("area", &Rect::area, DOC(dai, Rect, area))
        .def("empty", &Rect::empty, DOC(dai, Rect, empty))
        .def("contains", &Rect::contains, DOC(dai, Rect, contains))
        .def("isNormalized", &Rect::isNormalized, DOC(dai, Rect, isNormalized))
        .def("denormalize", &Rect::denormalize, py::arg("width"), py::arg("height"), DOC(dai, Rect, denormalize))
        .def("normalize", &Rect::normalize, py::arg("width"), py::arg("height"), DOC(dai, Rect, normalize))
        .def_readwrite("x", &Rect::x)
        .def_readwrite("y", &Rect::y)
        .def_readwrite("width", &Rect::width)
        .def_readwrite("height", &Rect::height);

    stereoPair.def(py::init<>())
        .def_readwrite("left", &StereoPair::left)
        .def_readwrite("right", &StereoPair::right)
        .def_readwrite("baseline", &StereoPair::baseline)
        .def_readwrite("isVertical", &StereoPair::isVertical)
        .def("__repr__", [](StereoPair& stereoPair) {
            std::stringstream stream;
            stream << stereoPair;
            return stream.str();
        });

    timestamp.def(py::init<>()).def_readwrite("sec", &Timestamp::sec).def_readwrite("nsec", &Timestamp::nsec).def("get", &Timestamp::get);

    point2f.def(py::init<>(), DOC(dai, Point2f, Point2f))
        .def(py::init<float, float>(), py::arg("x"), py::arg("y"), DOC(dai, Point2f, Point2f, 2))
        .def(py::init<float, float, bool>(), py::arg("x"), py::arg("y"), py::arg("normalized"), DOC(dai, Point2f, Point2f, 3))
        .def_readwrite("x", &Point2f::x)
        .def_readwrite("y", &Point2f::y)
        .def("isNormalized", &Point2f::isNormalized);

    point3f.def(py::init<>())
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Point3f::x)
        .def_readwrite("y", &Point3f::y)
        .def_readwrite("z", &Point3f::z);
    point3fRGBA.def(py::init<>())
        .def(py::init<float, float, float, int, int, int>())
        .def_readwrite("x", &Point3fRGBA::x)
        .def_readwrite("y", &Point3fRGBA::y)
        .def_readwrite("z", &Point3fRGBA::z)
        .def_readwrite("r", &Point3fRGBA::r)
        .def_readwrite("g", &Point3fRGBA::g)
        .def_readwrite("b", &Point3fRGBA::b)
        .def_readwrite("a", &Point3fRGBA::a);
    point3d.def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Point3d::x)
        .def_readwrite("y", &Point3d::y)
        .def_readwrite("z", &Point3d::z);
    quaterniond.def(py::init<>())
        .def(py::init<double, double, double, double>())
        .def_readwrite("qx", &Quaterniond::qx)
        .def_readwrite("qy", &Quaterniond::qy)
        .def_readwrite("qz", &Quaterniond::qz)
        .def_readwrite("qw", &Quaterniond::qw);
    size2f.def(py::init<>())
        .def(py::init<float, float>(), py::arg("width"), py::arg("height"))
        .def(py::init<float, float, bool>(), py::arg("width"), py::arg("height"), py::arg("normalized"))
        .def_readwrite("width", &Size2f::width)
        .def_readwrite("height", &Size2f::height)
        .def("isNormalized", &Size2f::isNormalized);

    // CameraBoardSocket enum bindings

    // Deprecated
    HEDLEY_DIAGNOSTIC_PUSH
    HEDLEY_DIAGNOSTIC_DISABLE_DEPRECATED
    cameraBoardSocket.value("AUTO", CameraBoardSocket::AUTO)
        .value("CAM_A", CameraBoardSocket::CAM_A)
        .value("CAM_B", CameraBoardSocket::CAM_B)
        .value("CAM_C", CameraBoardSocket::CAM_C)
        .value("CAM_D", CameraBoardSocket::CAM_D)
        .value("VERTICAL", CameraBoardSocket::VERTICAL)
        .value("CAM_E", CameraBoardSocket::CAM_E)
        .value("CAM_F", CameraBoardSocket::CAM_F)
        .value("CAM_G", CameraBoardSocket::CAM_G)
        .value("CAM_H", CameraBoardSocket::CAM_H)

        .value("RGB", CameraBoardSocket::RGB, "**Deprecated:** Use CAM_A or address camera by name instead")
        .value("LEFT", CameraBoardSocket::LEFT, "**Deprecated:** Use CAM_B or address camera by name instead")
        .value("RIGHT", CameraBoardSocket::RIGHT, "**Deprecated:** Use CAM_C or address camera by name instead")
        .value("CENTER", CameraBoardSocket::CENTER, "**Deprecated:** Use CAM_A or address camera by name instead")

        // Deprecated overriden
        .def_property_readonly_static("RGB",
                                      [](py::object) {
                                          PyErr_WarnEx(PyExc_DeprecationWarning, "RGB is deprecated, use CAM_A or address camera by name instead.", 1);
                                          return CameraBoardSocket::CAM_A;
                                      })
        .def_property_readonly_static("CENTER",
                                      [](py::object) {
                                          PyErr_WarnEx(PyExc_DeprecationWarning, "CENTER is deprecated, use CAM_A or address camera by name  instead.", 1);
                                          return CameraBoardSocket::CAM_A;
                                      })
        .def_property_readonly_static("LEFT",
                                      [](py::object) {
                                          PyErr_WarnEx(PyExc_DeprecationWarning, "LEFT is deprecated, use CAM_B or address camera by name  instead.", 1);
                                          return CameraBoardSocket::CAM_B;
                                      })
        .def_property_readonly_static("RIGHT", [](py::object) {
            PyErr_WarnEx(PyExc_DeprecationWarning, "RIGHT is deprecated, use CAM_C or address camera by name  instead.", 1);
            return CameraBoardSocket::CAM_C;
        });
    HEDLEY_DIAGNOSTIC_POP

    // CameraSensorType enum bindings
    cameraSensorType.value("COLOR", CameraSensorType::COLOR)
        .value("MONO", CameraSensorType::MONO)
        .value("TOF", CameraSensorType::TOF)
        .value("THERMAL", CameraSensorType::THERMAL);
    // ConnectionInterface enum bindings
    connectionInterface.value("USB", ConnectionInterface::USB).value("ETHERNET", ConnectionInterface::ETHERNET).value("WIFI", ConnectionInterface::WIFI);
    // CameraImageOrientation enum bindings
    cameraImageOrientation.value("AUTO", CameraImageOrientation::AUTO)
        .value("NORMAL", CameraImageOrientation::NORMAL)
        .value("HORIZONTAL_MIRROR", CameraImageOrientation::HORIZONTAL_MIRROR)
        .value("VERTICAL_FLIP", CameraImageOrientation::VERTICAL_FLIP)
        .value("ROTATE_180_DEG", CameraImageOrientation::ROTATE_180_DEG);

    // CameraFeatures
    cameraFeatures.def(py::init<>())
        .def_readwrite("socket", &CameraFeatures::socket)
        .def_readwrite("sensorName", &CameraFeatures::sensorName)
        .def_readwrite("width", &CameraFeatures::width)
        .def_readwrite("height", &CameraFeatures::height)
        .def_readwrite("orientation", &CameraFeatures::orientation)
        .def_readwrite("supportedTypes", &CameraFeatures::supportedTypes)
        .def_readwrite("hasAutofocus", &CameraFeatures::hasAutofocus)
        .def_readwrite("hasAutofocusIC", &CameraFeatures::hasAutofocusIC)
        .def_readwrite("name", &CameraFeatures::name)
        .def_readwrite("configs", &CameraFeatures::configs)
        .def_readwrite("calibrationResolution", &CameraFeatures::calibrationResolution)
        .def("__repr__", [](CameraFeatures& camera) {
            std::stringstream stream;
            stream << camera;
            return stream.str();
        });
    ;

    // CameraSensorConfig
    cameraSensorConfig.def(py::init<>())
        .def_readwrite("width", &CameraSensorConfig::width)
        .def_readwrite("height", &CameraSensorConfig::height)
        .def_readwrite("minFps", &CameraSensorConfig::minFps)
        .def_readwrite("maxFps", &CameraSensorConfig::maxFps)
        .def_readwrite("type", &CameraSensorConfig::type)
        .def_readwrite("fov", &CameraSensorConfig::fov)
        .def("__repr__", [](CameraSensorConfig& config) {
            std::stringstream stream;
            stream << config;
            return stream.str();
        });

    // MemoryInfo
    memoryInfo.def(py::init<>())
        .def_readwrite("remaining", &MemoryInfo::remaining)
        .def_readwrite("used", &MemoryInfo::used)
        .def_readwrite("total", &MemoryInfo::total);

    // ChipTemperature
    chipTemperature.def(py::init<>())
        .def_readwrite("css", &ChipTemperature::css)
        .def_readwrite("mss", &ChipTemperature::mss)
        .def_readwrite("upa", &ChipTemperature::upa)
        .def_readwrite("dss", &ChipTemperature::dss)
        .def_readwrite("average", &ChipTemperature::average);

    // ChipTemperatureS3
    chipTemperatureS3.def(py::init<>())
        .def_readwrite("css", &ChipTemperatureS3::css)
        .def_readwrite("mss", &ChipTemperatureS3::mss)
        .def_readwrite("nce", &ChipTemperatureS3::nce)
        .def_readwrite("soc", &ChipTemperatureS3::soc)
        .def_readwrite("average", &ChipTemperatureS3::average);

    // CpuUsage
    cpuUsage.def(py::init<>()).def_readwrite("average", &CpuUsage::average).def_readwrite("msTime", &CpuUsage::msTime);
    // CameraModel enum bindings
    cameraModel.value("Perspective", CameraModel::Perspective)
        .value("Fisheye", CameraModel::Fisheye)
        .value("Equirectangular", CameraModel::Equirectangular)
        .value("RadialDivision", CameraModel::RadialDivision);

    // StereoRectification
    stereoRectification.def(py::init<>())
        .def_readwrite("rectifiedRotationLeft", &StereoRectification::rectifiedRotationLeft)
        .def_readwrite("rectifiedRotationRight", &StereoRectification::rectifiedRotationRight)
        .def_readwrite("leftCameraSocket", &StereoRectification::leftCameraSocket)
        .def_readwrite("rightCameraSocket", &StereoRectification::rightCameraSocket);

    // Extrinsics
    extrinsics.def(py::init<>())
        .def_readwrite("rotationMatrix", &Extrinsics::rotationMatrix)
        .def_readwrite("translation", &Extrinsics::translation)
        .def_readwrite("specTranslation", &Extrinsics::specTranslation)
        .def_readwrite("toCameraSocket", &Extrinsics::toCameraSocket);

    // CameraInfo
    cameraInfo.def(py::init<>())
        .def_readwrite("width", &CameraInfo::width)
        .def_readwrite("height", &CameraInfo::height)
        .def_readwrite("intrinsicMatrix", &CameraInfo::intrinsicMatrix)
        .def_readwrite("distortionCoeff", &CameraInfo::distortionCoeff)
        .def_readwrite("extrinsics", &CameraInfo::extrinsics)
        .def_readwrite("cameraType", &CameraInfo::cameraType)
        .def_readwrite("specHfovDeg", &CameraInfo::specHfovDeg);

    // EepromData
    eepromData.def(py::init<>())
        .def_readwrite("version", &EepromData::version)
        .def_readwrite("boardCustom", &EepromData::boardCustom)
        .def_readwrite("boardName", &EepromData::boardName)
        .def_readwrite("boardRev", &EepromData::boardRev)
        .def_readwrite("boardConf", &EepromData::boardConf)
        .def_readwrite("hardwareConf", &EepromData::hardwareConf)
        .def_readwrite("productName", &EepromData::productName)
        .def_readwrite("batchName", &EepromData::batchName)
        .def_readwrite("deviceName", &EepromData::deviceName)
        .def_readwrite("batchTime", &EepromData::batchTime)
        .def_readwrite("boardOptions", &EepromData::boardOptions)
        .def_readwrite("cameraData", &EepromData::cameraData)
        .def_readwrite("stereoRectificationData", &EepromData::stereoRectificationData)
        .def_readwrite("imuExtrinsics", &EepromData::imuExtrinsics)
        .def_readwrite("miscellaneousData", &EepromData::miscellaneousData)
        .def_readwrite("housingExtrinsics", &EepromData::housingExtrinsics)
        .def_readwrite("stereoUseSpecTranslation", &EepromData::stereoUseSpecTranslation)
        .def_readwrite("stereoEnableDistortionCorrection", &EepromData::stereoEnableDistortionCorrection)
        .def_readwrite("verticalCameraSocket", &EepromData::verticalCameraSocket);
    // UsbSpeed
    usbSpeed.value("UNKNOWN", UsbSpeed::UNKNOWN)
        .value("LOW", UsbSpeed::LOW)
        .value("FULL", UsbSpeed::FULL)
        .value("HIGH", UsbSpeed::HIGH)
        .value("SUPER", UsbSpeed::SUPER)
        .value("SUPER_PLUS", UsbSpeed::SUPER_PLUS);

    // ProcessorType
    processorType.value("LEON_CSS", ProcessorType::LEON_CSS)
        .value("LEON_MSS", ProcessorType::LEON_MSS)
        .value("CPU", ProcessorType::CPU)
        .value("DSP", ProcessorType::DSP);

    detectionNetworkType.value("YOLO", DetectionNetworkType::YOLO).value("MOBILENET", DetectionNetworkType::MOBILENET);

    serializationType.value("LIBNOP", SerializationType::LIBNOP).value("JSON", SerializationType::JSON).value("JSON_MSGPACK", SerializationType::JSON_MSGPACK);

    detectionParserOptions.def_readwrite("nnFamily", &DetectionParserOptions::nnFamily)
        .def_readwrite("confidenceThreshold", &DetectionParserOptions::confidenceThreshold)
        .def_readwrite("classes", &DetectionParserOptions::classes)
        .def_readwrite("coordinates", &DetectionParserOptions::coordinates)
        .def_readwrite("anchors", &DetectionParserOptions::anchors)
        .def_readwrite("anchorMasks", &DetectionParserOptions::anchorMasks)
        .def_readwrite("iouThreshold", &DetectionParserOptions::iouThreshold);

    cameraExposureOffset.value("START", CameraExposureOffset::START).value("MIDDLE", CameraExposureOffset::MIDDLE).value("END", CameraExposureOffset::END);

    color.def(py::init<>())
        .def(py::init<float, float, float, float>(), py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a") = 1.0, DOC(dai, Color, Color))
        .def_readwrite("r", &Color::r)
        .def_readwrite("g", &Color::g)
        .def_readwrite("b", &Color::b)
        .def_readwrite("a", &Color::a);
    colormap.value("NONE", Colormap::NONE)
        .value("JET", Colormap::JET)
        .value("TURBO", Colormap::TURBO)
        .value("STEREO_JET", Colormap::STEREO_JET)
        .value("STEREO_TURBO", Colormap::STEREO_TURBO)
        // .value("AUTUMN", Colormap::AUTUMN)
        // .value("BONE", Colormap::BONE)
        // .value("WINTER", Colormap::WINTER)
        // .value("RAINBOW", Colormap::RAINBOW)
        // .value("OCEAN", Colormap::OCEAN)
        // .value("SUMMER", Colormap::SUMMER)
        // .value("SPRING", Colormap::SPRING)
        // .value("COOL", Colormap::COOL)
        // .value("HSV", Colormap::HSV)
        // .value("PINK", Colormap::PINK)
        // .value("HOT", Colormap::HOT)
        // .value("PARULA", Colormap::PARULA)
        // .value("MAGMA", Colormap::MAGMA)
        // .value("INFERNO", Colormap::INFERNO)
        // .value("PLASMA", Colormap::PLASMA)
        // .value("VIRIDIS", Colormap::VIRIDIS)
        // .value("CIVIDIS", Colormap::CIVIDIS)
        // .value("TWILIGHT", Colormap::TWILIGHT)
        // .value("TWILIGHT_SHIFTED", Colormap::TWILIGHT_SHIFTED)
        // .value("DEEPGREEN", Colormap::DEEPGREEN)
        ;

    frameEvent.value("NONE", FrameEvent::NONE).value("READOUT_START", FrameEvent::READOUT_START).value("READOUT_END", FrameEvent::READOUT_END);

    interpolation.value("BILINEAR", Interpolation::BILINEAR)
        .value("BICUBIC", Interpolation::BICUBIC)
        .value("NEAREST_NEIGHBOR", Interpolation::NEAREST_NEIGHBOR)
        .value("BYPASS", Interpolation::BYPASS)
        .value("DEFAULT", Interpolation::DEFAULT)
        .value("DEFAULT_DISPARITY_DEPTH", Interpolation::DEFAULT_DISPARITY_DEPTH);

    // backward compatibility
    // m.attr("node").attr("Warp").attr("Properties").attr("Interpolation") = interpolation;

    profilingData.def_readwrite("numBytesWritten", &ProfilingData::numBytesWritten, DOC(dai, ProfilingData, numBytesWritten))
        .def_readwrite("numBytesRead", &ProfilingData::numBytesRead, DOC(dai, ProfilingData, numBytesRead));
}
