set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Darwin)
set(VCPKG_OSX_ARCHITECTURES x86_64)
set(VCPKG_OSX_DEPLOYMENT_TARGET "11")

# Add ffmpeg after the shared libraries become relocatable
if(PORT MATCHES "libusb")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()
