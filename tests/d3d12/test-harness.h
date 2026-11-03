/*
* Minimal test harness for the NVRHI D3D12 backend.
*
* The harness is deliberately small: a static test registry, a handful of
* assertion macros, one lazily created D3D12 device, and helpers for shader
* compilation and GPU readback. It links only against NVRHI and the Windows
* SDK, so it can be added to any NVRHI checkout without extra dependencies.
*/

#pragma once

#include <nvrhi/d3d12.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nvrhi_test
{

    // Thrown by SKIP_IF to mark the current test as skipped. The reason is
    // printed by the runner.
    struct SkippedTest
    {
        std::string reason;
    };

    // Thrown by REQUIRE after the failure has been reported, so that the rest
    // of the test is not executed with an invalid precondition.
    struct FailedRequirement
    {
        std::string message;
    };

    struct TestCase
    {
        const char* name;
        void (*function)();
    };

    std::vector<TestCase>& tests();

    // Records one failed assertion and prints it immediately.
    void reportFailure(const char* file, int line, const std::string& message);

    // Registers a static test case. Instances are created during static
    // initialization in every translation unit that uses NVRHI_TEST_CASE.
    struct Registrar
    {
        Registrar(const char* name, void (*function)());
    };

    // Creates the process-wide D3D12 device. Returns false when no device can
    // be created; device tests then report themselves as skipped.
    bool createDevice();

    // Prints and clears the messages stored by the native D3D12 debug layer.
    void dumpDebugMessages();

    nvrhi::IDevice* device();
    nvrhi::d3d12::IDevice* d3d12Device();
    ID3D12Device* nativeDevice();
    nvrhi::IMessageCallback* messageCallback();
    int errorCount();
    struct CompiledShader
    {
        std::vector<uint8_t> bytecode;
    };

    // Compiles HLSL to DXIL with DXC. Every shader in this harness is compiled
    // as shader model 6.x DXIL.
    CompiledShader compileShader(const char* source, const char* entry, const char* profile);

    // True when the DXC compiler could be loaded and initialized at runtime.
    bool hasDxc();

    // Creates an upload-heap buffer and fills it with the supplied bytes.
    nvrhi::BufferHandle createUploadBuffer(nvrhi::IDevice* device, const void* data,
        uint64_t byteSize, const char* debugName);

    // Copies a buffer into a readback-heap buffer, waits for completion, and
    // returns the contents.
    std::vector<uint8_t> readBuffer(nvrhi::IDevice* device, nvrhi::IBuffer* buffer,
        uint64_t byteSize);

    // Copies one non-multisampled 2D texture subresource into a readback
    // staging texture and returns tightly packed rows.
    std::vector<uint8_t> readTexture2D(nvrhi::IDevice* device, nvrhi::ITexture* texture,
        uint32_t mipLevel = 0, uint32_t arraySlice = 0);

} // namespace nvrhi_test

#define NVRHI_TEST_CONCAT_INNER(a, b) a##b
#define NVRHI_TEST_CONCAT(a, b) NVRHI_TEST_CONCAT_INNER(a, b)

#define NVRHI_TEST_CASE(name)                                                                    \
    static void NVRHI_TEST_CONCAT(nvrhiTestFunction, __LINE__)();                                 \
    static ::nvrhi_test::Registrar NVRHI_TEST_CONCAT(nvrhiTestRegistrar, __LINE__)(               \
        name, &NVRHI_TEST_CONCAT(nvrhiTestFunction, __LINE__));                                   \
    static void NVRHI_TEST_CONCAT(nvrhiTestFunction, __LINE__)()

#define CHECK(expression)                                                                        \
    do                                                                                            \
    {                                                                                             \
        if (!(expression))                                                                        \
            ::nvrhi_test::reportFailure(__FILE__, __LINE__, #expression);                         \
    } while (0)

#define REQUIRE(expression)                                                                      \
    do                                                                                            \
    {                                                                                             \
        if (!(expression))                                                                        \
        {                                                                                         \
            ::nvrhi_test::reportFailure(__FILE__, __LINE__, #expression);                         \
            throw ::nvrhi_test::FailedRequirement{ #expression };                                 \
        }                                                                                         \
    } while (0)

#define CHECK_EQ(actual, expected)                                                               \
    do                                                                                            \
    {                                                                                             \
        const auto& nvrhiTestActual = (actual);                                                   \
        const auto& nvrhiTestExpected = (expected);                                               \
        if (!(nvrhiTestActual == nvrhiTestExpected))                                              \
            ::nvrhi_test::reportFailure(__FILE__, __LINE__,                                       \
                std::string(#actual " == " #expected));                                           \
    } while (0)

#define CHECK_NE(actual, expected)                                                               \
    do                                                                                            \
    {                                                                                             \
        const auto& nvrhiTestActual = (actual);                                                   \
        const auto& nvrhiTestExpected = (expected);                                               \
        if (nvrhiTestActual == nvrhiTestExpected)                                                 \
            ::nvrhi_test::reportFailure(__FILE__, __LINE__,                                       \
                std::string(#actual " != " #expected));                                           \
    } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                                  \
    do                                                                                            \
    {                                                                                             \
        const double nvrhiTestActual = double(actual);                                            \
        const double nvrhiTestExpected = double(expected);                                        \
        if (!(std::abs(nvrhiTestActual - nvrhiTestExpected) <= double(tolerance)))                \
            ::nvrhi_test::reportFailure(__FILE__, __LINE__,                                       \
                std::string(#actual " is near " #expected));                                      \
    } while (0)

#define SKIP_IF(condition, reason)                                                               \
    do                                                                                            \
    {                                                                                             \
        if (condition)                                                                            \
            throw ::nvrhi_test::SkippedTest{ reason };                                            \
    } while (0)

#define REQUIRE_DEVICE()                                                                         \
    SKIP_IF(::nvrhi_test::device() == nullptr, "D3D12 device is unavailable")
