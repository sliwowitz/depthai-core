
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO luxonis/libusb
    REF b7e4548958325b18feb73977163ad44398099534
    SHA512 b0fd00fe623a6fe527046002d94201cfe2dffe589bd77aa88473f25960e3ce7206a90290ee4fe66fa2e91766c5f6585379ba2b008e9fc7b33da43763127a4ef2
    HEAD_REF main
)
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        WITH_UDEV=OFF
        # Build shared libs by default to not cause licensing issues
        BUILD_SHARED_LIBS=ON
)

vcpkg_cmake_install()

vcpkg_fixup_pkgconfig()
