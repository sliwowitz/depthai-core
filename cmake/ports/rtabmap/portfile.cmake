vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO introlab/rtabmap
    REF 08f031e11c45589fc2b68440383a3e40982dc06f
    SHA512 3a0c97ac2061a38c01c2b9fb95057061f38e79ce8500b73010fdf6ba3547196960da749ea5829571c227f8379a4c8482be3b7d8df68984e429df14a8020a24a3
    HEAD_REF master
    PATCHES
        apple.patch
        fix_link.patch
        link-keywords.patch
        multi-definition.patch
        rtabmap-res-tool.patch
        gklib.patch
        lz4.patch
        ceres_typo.patch
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        gui         WITH_QT
        k4w2        WITH_K4W2
        octomap     WITH_OCTOMAP
        openni2     WITH_OPENNI2
        realsense2  WITH_REALSENSE2
        tools       BUILD_APP
        tools       BUILD_TOOLS
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        ${FEATURE_OPTIONS}
        -DBUILD_TOOLS=OFF
        -DBUILD_APP=OFF
        -DBUILD_AS_BUNDLE=OFF
        -DBUILD_EXAMPLES=OFF
        -DRTABMAP_QT_VERSION=6
        "-DRTABMAP_RES_TOOL=${CURRENT_HOST_INSTALLED_DIR}/tools/rtabmap-res-tool/rtabmap-res_tool${VCPKG_TARGET_EXECUTABLE_SUFFIX}"
        -DWITH_ALICE_VISION=OFF
        -DWITH_CCCORELIB=OFF
        -DWITH_CERES=ON
        -DWITH_CPUTSDF=OFF
        -DWITH_CVSBA=OFF
        -DWITH_DC1394=OFF
        -DWITH_DEPTHAI=OFF
        -DWITH_DVO=OFF
        -DWITH_FASTCV=OFF
        -DWITH_FLYCAPTURE2=OFF
        -DWITH_FOVIS=OFF
        -DWITH_FREENECT=OFF
        -DWITH_FREENECT2=OFF
        -DWITH_G2O=ON
        -DWITH_GTSAM=OFF
        -DWITH_K4A=OFF
        -DWITH_LOAM=OFF
        -DWITH_MADGWICK=OFF
        -DWITH_MSCKF_VIO=OFF
        -DWITH_MYNTEYE=OFF
        -DWITH_OKVIS=OFF
        -DWITH_OPENCHISEL=OFF
        -DWITH_OPENVINS=OFF
        -DWITH_ORB_OCTREE=ON   # GPLv3
        -DWITH_ORB_SLAM=OFF
        -DWITH_PDAL=OFF
        -DWITH_POINTMATCHER=OFF
        -DWITH_PYTHON_THREADING=OFF
        -DWITH_PYTHON=OFF
        -DWITH_REALSENSE_SLAM=OFF
        -DWITH_REALSENSE=OFF
        -DWITH_TORCH=OFF
        -DWITH_VERTIGO=OFF
        -DWITH_VINS=OFF
        -DWITH_VISO2=OFF
        -DWITH_ZED=OFF
        -DWITH_ZEDOC=OFF
)

vcpkg_cmake_install()

if(VCPKG_TARGET_IS_WINDOWS)
    vcpkg_cmake_config_fixup(CONFIG_PATH CMake)
else()
    vcpkg_cmake_config_fixup(CONFIG_PATH lib/rtabmap-0.21)
endif()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

if("tools" IN_LIST FEATURES)
  vcpkg_copy_tools(
    TOOL_NAMES
        rtabmap-camera
        rtabmap-console
        rtabmap-detectMoreLoopClosures
        rtabmap-export
        rtabmap-extractObject
        rtabmap-info
        rtabmap-kitti_dataset
        rtabmap-recovery
        rtabmap-report
        rtabmap-reprocess
        rtabmap-rgbd_dataset
        rtabmap-euroc_dataset
        rtabmap-cleanupLocalGrids
        rtabmap-globalBundleAdjustment
    AUTO_CLEAN
  )
  if("gui" IN_LIST FEATURES)
    vcpkg_copy_tools(
        TOOL_NAMES
            rtabmap
            rtabmap-calibration
            rtabmap-databaseViewer
            rtabmap-dataRecorder
            rtabmap-odometryViewer
            rtabmap-rgbd_camera
        AUTO_CLEAN
    )
    file(COPY "${CURRENT_INSTALLED_DIR}/tools/Qt6/bin/qt.conf" DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")
    vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/tools/${PORT}/qt.conf" "./../../../" "./../../")
  endif()
endif()

vcpkg_install_copyright(
    COMMENT [[
The RTAB-Map main license is BSD-3-Clause, but some parts of the
source code are under other licenses possibly including GPL-3.0-only.
]]
    FILE_LIST "${SOURCE_PATH}/LICENSE" 
)
