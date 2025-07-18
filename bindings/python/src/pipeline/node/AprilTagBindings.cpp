#include "Common.hpp"
#include "NodeBindings.hpp"
#include "depthai/pipeline/Node.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/AprilTag.hpp"

void bind_apriltag(pybind11::module& m, void* pCallstack) {
    using namespace dai;
    using namespace dai::node;

    // Node and Properties declare upfront
    py::class_<AprilTagProperties> aprilTagProperties(m, "AprilTagProperties", DOC(dai, AprilTagProperties));
    auto aprilTag = ADD_NODE(AprilTag);

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

    // Properties
    aprilTagProperties.def_readwrite("initialConfig", &AprilTagProperties::initialConfig, DOC(dai, AprilTagProperties, initialConfig))
        .def_readwrite("inputConfigSync", &AprilTagProperties::inputConfigSync, DOC(dai, AprilTagProperties, inputConfigSync))
        .def_readwrite("numThreads", &AprilTagProperties::numThreads, DOC(dai, AprilTagProperties, numThreads));
    // Node
    aprilTag.def_readonly("inputConfig", &AprilTag::inputConfig, DOC(dai, node, AprilTag, inputConfig))
        .def_readonly("inputImage", &AprilTag::inputImage, DOC(dai, node, AprilTag, inputImage))
        .def_readonly("out", &AprilTag::out, DOC(dai, node, AprilTag, out))
        .def_readonly("passthroughInputImage", &AprilTag::passthroughInputImage, DOC(dai, node, AprilTag, passthroughInputImage))
        .def_readonly("initialConfig", &AprilTag::initialConfig, DOC(dai, node, AprilTag, initialConfig))
        .def("setWaitForConfigInput", &AprilTag::setWaitForConfigInput, py::arg("wait"), DOC(dai, node, AprilTag, setWaitForConfigInput))
        .def("getWaitForConfigInput", &AprilTag::getWaitForConfigInput, DOC(dai, node, AprilTag, getWaitForConfigInput))
        .def("runOnHost", &AprilTag::runOnHost, DOC(dai, node, AprilTag, runOnHost))
        .def("setRunOnHost", &AprilTag::setRunOnHost, DOC(dai, node, AprilTag, setRunOnHost))
        .def("setNumThreads", &AprilTag::setNumThreads, py::arg("numThreads"), DOC(dai, node, AprilTag, setNumThreads))
        .def("getNumThreads", &AprilTag::getNumThreads, DOC(dai, node, AprilTag, getNumThreads));
    daiNodeModule.attr("AprilTag").attr("Properties") = aprilTagProperties;
}
