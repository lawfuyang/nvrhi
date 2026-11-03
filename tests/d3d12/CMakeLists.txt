add_executable(nvrhi_d3d12_tests
    test-harness.h
    test-harness.cpp
    core-tests.cpp
    device-tests.cpp
    pipeline-tests.cpp)

target_link_libraries(nvrhi_d3d12_tests PRIVATE
    nvrhi
    ${nvrhi_d3d12_target}
    dxgi)

target_include_directories(nvrhi_d3d12_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# DXC is required: every test shader is compiled to DXIL (shader model 6.x).
# Recent Windows SDKs ship the compiler, its import library, dxcapi.h and the
# DXIL signer, so no external dependency is needed.
# $ENV{ProgramFiles(x86)} is not parseable by CMake, so derive it from
# ProgramW6432 when available.
if(DEFINED ENV{ProgramW6432})
    set(_nvrhi_sdk_root "$ENV{ProgramW6432} (x86)/Windows Kits/10")
else()
    set(_nvrhi_sdk_root "$ENV{ProgramFiles}/Windows Kits/10")
endif()
set(_nvrhi_sdk_versions "")

if(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
    list(APPEND _nvrhi_sdk_versions "${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
else()
    file(GLOB _nvrhi_sdk_dirs LIST_DIRECTORIES true "${_nvrhi_sdk_root}/bin/10.*")
    foreach(_nvrhi_sdk_dir IN LISTS _nvrhi_sdk_dirs)
        get_filename_component(_nvrhi_sdk_version "${_nvrhi_sdk_dir}" NAME)
        list(APPEND _nvrhi_sdk_versions "${_nvrhi_sdk_version}")
    endforeach()
    list(SORT _nvrhi_sdk_versions)
    list(REVERSE _nvrhi_sdk_versions)
endif()

set(NVRHI_DXC_VERSION "")

# find_* skips the search when the result variable is already defined, so clear
# both the normal and the cached values before probing.
unset(NVRHI_DXC_DLL)
unset(NVRHI_DXIL_DLL)
unset(NVRHI_DXC_INCLUDE_DIR)
unset(NVRHI_DXCOMPILER_LIB)
unset(NVRHI_DXC_DLL CACHE)
unset(NVRHI_DXIL_DLL CACHE)
unset(NVRHI_DXC_INCLUDE_DIR CACHE)
unset(NVRHI_DXCOMPILER_LIB CACHE)

foreach(_nvrhi_sdk_version IN LISTS _nvrhi_sdk_versions)
    if(NOT NVRHI_DXC_DLL)
        find_file(NVRHI_DXC_DLL
            NAMES dxcompiler.dll
            PATHS "${_nvrhi_sdk_root}/bin/${_nvrhi_sdk_version}/x64"
            NO_DEFAULT_PATH)
    endif()
    if(NOT NVRHI_DXIL_DLL)
        find_file(NVRHI_DXIL_DLL
            NAMES dxil.dll
            PATHS "${_nvrhi_sdk_root}/bin/${_nvrhi_sdk_version}/x64"
            NO_DEFAULT_PATH)
    endif()
    if(NOT NVRHI_DXC_INCLUDE_DIR)
        find_path(NVRHI_DXC_INCLUDE_DIR
            NAMES dxcapi.h
            PATHS "${_nvrhi_sdk_root}/Include/${_nvrhi_sdk_version}/um"
            NO_DEFAULT_PATH)
    endif()
    if(NOT NVRHI_DXCOMPILER_LIB)
        find_library(NVRHI_DXCOMPILER_LIB
            NAMES dxcompiler
            PATHS "${_nvrhi_sdk_root}/Lib/${_nvrhi_sdk_version}/um/x64"
            NO_DEFAULT_PATH)
    endif()

    if(NVRHI_DXC_DLL AND NVRHI_DXIL_DLL AND NVRHI_DXC_INCLUDE_DIR AND NVRHI_DXCOMPILER_LIB)
        set(NVRHI_DXC_VERSION "${_nvrhi_sdk_version}")
        break()
    endif()
endforeach()

if(NVRHI_DXC_DLL AND NVRHI_DXIL_DLL AND NVRHI_DXC_INCLUDE_DIR AND NVRHI_DXCOMPILER_LIB)
    target_compile_definitions(nvrhi_d3d12_tests PRIVATE NVRHI_TEST_WITH_DXC=1)
    target_include_directories(nvrhi_d3d12_tests PRIVATE "${NVRHI_DXC_INCLUDE_DIR}")
    target_link_libraries(nvrhi_d3d12_tests PRIVATE "${NVRHI_DXCOMPILER_LIB}")

    add_custom_command(TARGET nvrhi_d3d12_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVRHI_DXC_DLL}" "$<TARGET_FILE_DIR:nvrhi_d3d12_tests>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NVRHI_DXIL_DLL}" "$<TARGET_FILE_DIR:nvrhi_d3d12_tests>")

    message(STATUS "nvrhi_d3d12_tests: DXC from Windows SDK ${NVRHI_DXC_VERSION}")
else()
    message(FATAL_ERROR "nvrhi_d3d12_tests requires DXC (dxcompiler.dll, dxil.dll, dxcapi.h and dxcompiler.lib). Install a Windows SDK that provides them.")
endif()

set_target_properties(nvrhi_d3d12_tests PROPERTIES FOLDER "NVRHI/Tests")

add_test(NAME d3d12_tests COMMAND nvrhi_d3d12_tests)
set_tests_properties(d3d12_tests PROPERTIES TIMEOUT 120)
