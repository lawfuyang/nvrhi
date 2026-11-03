#include "test-harness.h"

#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#if NVRHI_TEST_WITH_DXC
#include <dxcapi.h>
#endif

#include <cstdio>
#include <exception>

namespace nvrhi_test
{
    namespace
    {
        using Microsoft::WRL::ComPtr;

        struct MessageCollector : nvrhi::IMessageCallback
        {
            int errors = 0;

            void message(nvrhi::MessageSeverity severity, const char* text) override
            {
                if (severity == nvrhi::MessageSeverity::Error ||
                    severity == nvrhi::MessageSeverity::Fatal)
                {
                    ++errors;
                    std::printf("    NVRHI error: %s\n", text);
                }
                else if (severity == nvrhi::MessageSeverity::Warning)
                {
                    std::printf("    NVRHI warning: %s\n", text);
                }
            }
        };

        std::vector<TestCase>& registry();

        std::vector<TestCase>& registry()
        {
            static std::vector<TestCase> instance;
            return instance;
        }

        int g_TotalFailures = 0;

        ComPtr<ID3D12Device> g_NativeDevice;
        ComPtr<ID3D12CommandQueue> g_GraphicsQueue;
        ComPtr<ID3D12CommandQueue> g_ComputeQueue;
        ComPtr<ID3D12CommandQueue> g_CopyQueue;
        nvrhi::DeviceHandle g_Device;
        nvrhi::d3d12::IDevice* g_D3D12Device = nullptr;
        MessageCollector g_Messages;

        ComPtr<IDXGIAdapter1> findWarpAdapter()
        {
            ComPtr<IDXGIFactory4> factory;
            if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
                return nullptr;

            ComPtr<IDXGIAdapter1> adapter;
            factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
            return adapter;
        }

        ComPtr<ID3D12CommandQueue> createQueue(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type)
        {
            D3D12_COMMAND_QUEUE_DESC desc = {};
            desc.Type = type;
            desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
            desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
            desc.NodeMask = 0;

            ComPtr<ID3D12CommandQueue> queue;
            device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue));
            return queue;
        }

    } // namespace

    std::vector<TestCase>& tests()
    {
        return registry();
    }

    void reportFailure(const char* file, int line, const std::string& message)
    {
        ++g_TotalFailures;
        std::printf("    %s:%d: check failed: %s\n", file, line, message.c_str());
    }

    Registrar::Registrar(const char* name, void (*function)())
    {
        registry().push_back(TestCase{ name, function });
    }

    bool createDevice()
    {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
            debug->EnableDebugLayer();

        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_NativeDevice))))
        {
            ComPtr<IDXGIAdapter1> warp = findWarpAdapter();
            if (warp)
                D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_NativeDevice));
        }

        if (!g_NativeDevice)
            return false;

        g_GraphicsQueue = createQueue(g_NativeDevice.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
        g_ComputeQueue = createQueue(g_NativeDevice.Get(), D3D12_COMMAND_LIST_TYPE_COMPUTE);
        g_CopyQueue = createQueue(g_NativeDevice.Get(), D3D12_COMMAND_LIST_TYPE_COPY);

        if (!g_GraphicsQueue)
            return false;

        nvrhi::d3d12::DeviceDesc desc;
        desc.pDevice = g_NativeDevice.Get();
        desc.pGraphicsCommandQueue = g_GraphicsQueue.Get();
        desc.pComputeCommandQueue = g_ComputeQueue.Get();
        desc.pCopyCommandQueue = g_CopyQueue.Get();
        desc.errorCB = &g_Messages;

        g_Device = nvrhi::d3d12::createDevice(desc);
        if (!g_Device)
            return false;

        g_D3D12Device = static_cast<nvrhi::d3d12::IDevice*>(g_Device.Get());
        return true;
    }

    nvrhi::IDevice* device()
    {
        return g_Device.Get();
    }
    nvrhi::d3d12::IDevice* d3d12Device()
    {
        return g_D3D12Device;
    }

    ID3D12Device* nativeDevice()
    {
        return g_NativeDevice.Get();
    }

    nvrhi::IMessageCallback* messageCallback()
    {
        return &g_Messages;
    }

    void dumpDebugMessages()
    {
        if (!g_NativeDevice)
            return;

        ComPtr<ID3D12InfoQueue> infoQueue;
        if (FAILED(g_NativeDevice.As(&infoQueue)))
            return;

        const UINT64 count = infoQueue->GetNumStoredMessages();
        for (UINT64 index = 0; index < count; ++index)
        {
            SIZE_T length = 0;
            if (FAILED(infoQueue->GetMessage(index, nullptr, &length)) || length == 0)
                continue;

            std::vector<uint8_t> storage(length);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (FAILED(infoQueue->GetMessage(index, message, &length)))
                continue;

            std::printf("    D3D12 debug: %s\n", message->pDescription);
        }

        infoQueue->ClearStoredMessages();
    }

    int errorCount()
    {
        return g_Messages.errors;
    }

    bool hasDxc()
    {
#if NVRHI_TEST_WITH_DXC
        static const bool available = []()
        {
            ComPtr<IDxcCompiler3> compiler;
            return SUCCEEDED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)));
        }();
        return available;
#else
        return false;
#endif
    }

    CompiledShader compileShader(const char* source, const char* entry,
        const char* profile)
    {
        REQUIRE(hasDxc());

#if NVRHI_TEST_WITH_DXC
        ComPtr<IDxcUtils> utils;
        REQUIRE(SUCCEEDED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils))));

        ComPtr<IDxcCompiler3> compiler;
        REQUIRE(SUCCEEDED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler))));

        const UINT32 sourceSize = UINT32(std::strlen(source));
        ComPtr<IDxcBlobEncoding> sourceBlob;
        REQUIRE(SUCCEEDED(utils->CreateBlobFromPinned(source, sourceSize,
            DXC_CP_UTF8, &sourceBlob)));

        const auto toWide = [](const char* text)
        {
            const size_t length = std::strlen(text);
            return std::wstring(text, text + length);
        };

        const std::wstring entryWide = toWide(entry != nullptr ? entry : "");
        const std::wstring profileWide = toWide(profile != nullptr ? profile : "");

        std::vector<std::wstring> argumentStorage = {
            L"-HV", L"2021",
            L"-T", profileWide,
        };
        if (!entryWide.empty())
        {
            argumentStorage.push_back(L"-E");
            argumentStorage.push_back(entryWide);
        }

        std::vector<LPCWSTR> arguments;
        arguments.reserve(argumentStorage.size());
        for (const std::wstring& argument : argumentStorage)
            arguments.push_back(argument.c_str());

        DxcBuffer sourceBuffer = {};
        sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
        sourceBuffer.Size = sourceBlob->GetBufferSize();
        sourceBuffer.Encoding = DXC_CP_UTF8;

        ComPtr<IDxcResult> result;
        REQUIRE(SUCCEEDED(compiler->Compile(&sourceBuffer, arguments.data(),
            UINT32(arguments.size()), nullptr, IID_PPV_ARGS(&result))));

        HRESULT status = E_FAIL;
        REQUIRE(SUCCEEDED(result->GetStatus(&status)));

        if (FAILED(status))
        {
            std::string text = "DXIL shader compilation failed";
            ComPtr<IDxcBlobUtf8> diagnostics;
            if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS,
                IID_PPV_ARGS(&diagnostics), nullptr)) && diagnostics != nullptr)
            {
                text.assign(diagnostics->GetStringPointer(),
                    diagnostics->GetStringLength());
            }
            reportFailure(__FILE__, __LINE__, text);
            throw FailedRequirement{ text };
        }

        ComPtr<IDxcBlob> object;
        REQUIRE(SUCCEEDED(result->GetOutput(DXC_OUT_OBJECT,
            IID_PPV_ARGS(&object), nullptr)));
        REQUIRE(object != nullptr);

        CompiledShader compiled;
        const uint8_t* bytes = static_cast<const uint8_t*>(object->GetBufferPointer());
        compiled.bytecode.assign(bytes, bytes + object->GetBufferSize());
        return compiled;
#else
        std::string text = "DXC is required but was not enabled at build time (source ";
        text += std::to_string(std::strlen(source));
        text += " bytes, entry ";
        text += (entry != nullptr && entry[0] != '\0') ? entry : "<none>";
        text += ", profile ";
        text += (profile != nullptr) ? profile : "<none>";
        text += ")";
        reportFailure(__FILE__, __LINE__, text);
        throw FailedRequirement{ text };
#endif
    }

    nvrhi::BufferHandle createUploadBuffer(nvrhi::IDevice* testDevice, const void* data,
        uint64_t byteSize, const char* debugName)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = byteSize;
        desc.debugName = debugName;
        desc.cpuAccess = nvrhi::CpuAccessMode::Write;

        nvrhi::BufferHandle buffer = testDevice->createBuffer(desc);
        REQUIRE(buffer != nullptr);

        void* mapped = testDevice->mapBuffer(buffer, nvrhi::CpuAccessMode::Write);
        REQUIRE(mapped != nullptr);
        std::memcpy(mapped, data, static_cast<size_t>(byteSize));
        testDevice->unmapBuffer(buffer);

        return buffer;
    }

    std::vector<uint8_t> readBuffer(nvrhi::IDevice* testDevice, nvrhi::IBuffer* buffer,
        uint64_t byteSize)
    {
        nvrhi::BufferDesc readbackDesc;
        readbackDesc.byteSize = byteSize;
        readbackDesc.debugName = "Readback buffer";
        readbackDesc.cpuAccess = nvrhi::CpuAccessMode::Read;

        nvrhi::BufferHandle readback = testDevice->createBuffer(readbackDesc);
        REQUIRE(readback != nullptr);

        nvrhi::CommandListHandle commandList = testDevice->createCommandList();
        REQUIRE(commandList != nullptr);
        commandList->open();
        commandList->copyBuffer(readback, 0, buffer, 0, byteSize);
        commandList->close();
        testDevice->executeCommandList(commandList);
        REQUIRE(testDevice->waitForIdle());

        void* mapped = testDevice->mapBuffer(readback, nvrhi::CpuAccessMode::Read);
        REQUIRE(mapped != nullptr);

        std::vector<uint8_t> result(static_cast<size_t>(byteSize));
        std::memcpy(result.data(), mapped, static_cast<size_t>(byteSize));
        testDevice->unmapBuffer(readback);

        return result;
    }

    std::vector<uint8_t> readTexture2D(nvrhi::IDevice* testDevice, nvrhi::ITexture* texture,
        uint32_t mipLevel, uint32_t arraySlice)
    {
        const nvrhi::TextureDesc& desc = texture->getDesc();
        REQUIRE(desc.sampleCount == 1);

        nvrhi::TextureDesc stagingDesc = desc;
        stagingDesc.debugName = "Readback staging texture";

        nvrhi::StagingTextureHandle staging =
            testDevice->createStagingTexture(stagingDesc, nvrhi::CpuAccessMode::Read);
        REQUIRE(staging != nullptr);

        nvrhi::TextureSlice slice = nvrhi::TextureSlice()
            .setMipLevel(mipLevel)
            .setArraySlice(arraySlice);
        const nvrhi::TextureSlice resolved = slice.resolve(desc);
        REQUIRE(resolved.depth == 1);

        nvrhi::CommandListHandle commandList = testDevice->createCommandList();
        REQUIRE(commandList != nullptr);
        commandList->open();
        commandList->copyTexture(staging, resolved, texture, resolved);
        commandList->close();
        testDevice->executeCommandList(commandList);
        REQUIRE(testDevice->waitForIdle());

        size_t rowPitch = 0;
        void* mapped = testDevice->mapStagingTexture(staging, resolved,
            nvrhi::CpuAccessMode::Read, &rowPitch);
        REQUIRE(mapped != nullptr);

        const nvrhi::FormatInfo& formatInfo = nvrhi::getFormatInfo(desc.format);
        const size_t rowBytes = size_t(resolved.width / formatInfo.blockSize) * formatInfo.bytesPerBlock;
        std::vector<uint8_t> result(rowBytes * resolved.height);
        for (uint32_t row = 0; row < resolved.height; ++row)
        {
            std::memcpy(result.data() + row * rowBytes,
                static_cast<const uint8_t*>(mapped) + row * rowPitch, rowBytes);
        }

        testDevice->unmapStagingTexture(staging);
        return result;
    }

} // namespace nvrhi_test

int main()
{
    using namespace nvrhi_test;

    std::setvbuf(stdout, nullptr, _IONBF, 0);

    const bool hasDevice = createDevice();
    std::printf("NVRHI D3D12 test harness: %s\n\n",
        hasDevice ? "device created" : "no D3D12 device, GPU tests will be skipped");

    int passed = 0;
    int failed = 0;
    int skipped = 0;

    for (const TestCase& test : tests())
    {
        std::printf("[ RUN      ] %s\n", test.name);

        const int failuresBefore = g_TotalFailures;
        const int errorsBefore = errorCount();

        try
        {
            test.function();
        }
        catch (const SkippedTest& skippedTest)
        {
            std::printf("[ SKIPPED  ] %s (%s)\n", test.name, skippedTest.reason.c_str());
            ++skipped;
            continue;
        }
        catch (const FailedRequirement&)
        {
        }
        catch (const std::exception& exception)
        {
            reportFailure(__FILE__, __LINE__, exception.what());
        }

        if (g_TotalFailures == failuresBefore && errorCount() != errorsBefore)
            reportFailure(__FILE__, __LINE__, "NVRHI reported one or more errors");

        if (g_TotalFailures != failuresBefore)
        {
            std::printf("[ FAILED   ] %s\n", test.name);
            dumpDebugMessages();
            ++failed;
        }
        else
        {
            std::printf("[ PASSED   ] %s\n", test.name);
            ++passed;
        }
    }

    std::printf("\n%d passed, %d failed, %d skipped\n", passed, failed, skipped);
    return failed == 0 ? 0 : 1;
}
