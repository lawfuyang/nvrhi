#include "test-harness.h"

#include <nvrhi/common/misc.h>
#include <nvrhi/utils.h>

#include <wrl/client.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
    const char* const c_FullscreenVertexShader = R"(
struct VSOutput
{
    float4 position : SV_Position;
};

VSOutput main(uint vertexID : SV_VertexID)
{
    float2 positions[3] =
    {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };

    VSOutput output;
    output.position = float4(positions[vertexID], 0.5, 1.0);
    return output;
}
)";

    const char* const c_SolidColorPixelShader = R"(
float4 main() : SV_Target
{
    return float4(0.25, 0.5, 0.75, 1.0);
}
)";

    const char* const c_GreenPixelShader = R"(
float4 main() : SV_Target
{
    return float4(0.0, 1.0, 0.0, 1.0);
}
)";

    const char* const c_RedPixelShader = R"(
float4 main() : SV_Target
{
    return float4(1.0, 0.0, 0.0, 1.0);
}
)";

    const char* const c_ClearComputeShader = R"(
RWStructuredBuffer<uint> gOutput : register(u0);

[numthreads(32, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    gOutput[threadID.x] = threadID.x * 3 + 1;
}
)";

    const char* const c_PushConstantComputeShader = R"(
RWStructuredBuffer<uint> gOutput : register(u0);

cbuffer Parameters : register(b0)
{
    uint gMultiplier;
};

[numthreads(32, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    gOutput[threadID.x] = threadID.x * gMultiplier;
}
)";

    const char* const c_MeshShader = R"(
struct MeshVertex
{
    float4 position : SV_Position;
};

[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void main(out vertices MeshVertex vertices[3], out indices uint3 triangles[1])
{
    SetMeshOutputCounts(3, 1);

    vertices[0].position = float4(-1.0, -1.0, 0.5, 1.0);
    vertices[1].position = float4( 3.0, -1.0, 0.5, 1.0);
    vertices[2].position = float4(-1.0,  3.0, 0.5, 1.0);

    triangles[0] = uint3(0, 1, 2);
}
)";

    const char* const c_RayTracingLibrary = R"(
struct Payload
{
    uint value;
};

RWStructuredBuffer<uint> gOutput : register(u0);

[shader("raygeneration")]
void raygen()
{
    gOutput[0] = 42;
}

[shader("miss")]
void miss(inout Payload payload)
{
    payload.value = 1;
}

[shader("closesthit")]
void closest(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    payload.value = 2;
}

[shader("callable")]
void callable(inout Payload payload)
{
    payload.value = 3;
}
)";

    nvrhi::ShaderHandle createShader(nvrhi::IDevice* device, nvrhi::ShaderType type,
        const char* source, const char* profile, const char* debugName)
    {
        const nvrhi_test::CompiledShader compiled =
            nvrhi_test::compileShader(source, "main", profile);
        return device->createShader(
            nvrhi::ShaderDesc().setShaderType(type).setDebugName(debugName),
            compiled.bytecode.data(), compiled.bytecode.size());
    }

    struct FullscreenPipeline
    {
        nvrhi::ShaderHandle vertexShader;
        nvrhi::ShaderHandle pixelShader;
        nvrhi::GraphicsPipelineHandle pipeline;
    };

    FullscreenPipeline createFullscreenPipeline(nvrhi::IDevice* device,
        const nvrhi::FramebufferInfo& framebufferInfo, const char* pixelShaderSource,
        const nvrhi::DepthStencilState& depthState)
    {
        FullscreenPipeline result;
        result.vertexShader = createShader(device, nvrhi::ShaderType::Vertex,
            c_FullscreenVertexShader, "vs_6_0", "Fullscreen vertex shader");
        result.pixelShader = createShader(device, nvrhi::ShaderType::Pixel,
            pixelShaderSource, "ps_6_0", "Fullscreen pixel shader");

        nvrhi::GraphicsPipelineDesc desc;
        desc.setPrimType(nvrhi::PrimitiveType::TriangleList)
            .setVertexShader(result.vertexShader)
            .setPixelShader(result.pixelShader);
        desc.renderState.rasterState.setCullNone();
        desc.renderState.depthStencilState = depthState;

        result.pipeline = device->createGraphicsPipeline(desc, framebufferInfo);
        REQUIRE(result.pipeline != nullptr);
        return result;
    }

    nvrhi::GraphicsState makeFullscreenState(nvrhi::IGraphicsPipeline* pipeline,
        nvrhi::IFramebuffer* framebuffer, uint32_t width, uint32_t height)
    {
        nvrhi::GraphicsState state;
        state.setPipeline(pipeline)
            .setFramebuffer(framebuffer)
            .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(
                nvrhi::Viewport(float(width), float(height))));
        return state;
    }

    void checkPixel(const std::vector<uint8_t>& pixels, uint32_t width,
        uint32_t x, uint32_t y, uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha)
    {
        const size_t offset = (size_t(y) * width + x) * 4;
        REQUIRE(pixels.size() >= offset + 4);
        CHECK_NEAR(pixels[offset + 0], red, 1);
        CHECK_NEAR(pixels[offset + 1], green, 1);
        CHECK_NEAR(pixels[offset + 2], blue, 1);
        CHECK_NEAR(pixels[offset + 3], alpha, 1);
    }

    void checkAllPixels(const std::vector<uint8_t>& pixels, uint32_t width, uint32_t height,
        uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
                checkPixel(pixels, width, x, y, red, green, blue, alpha);
        }
    }
}

NVRHI_TEST_CASE("shaders preserve bytecode and their description")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    const nvrhi_test::CompiledShader compiled =
        nvrhi_test::compileShader(c_SolidColorPixelShader, "main", "ps_6_0");
    REQUIRE(!compiled.bytecode.empty());

    nvrhi::ShaderDesc desc;
    desc.setShaderType(nvrhi::ShaderType::Pixel)
        .setDebugName("Solid color")
        .setEntryName("main");

    nvrhi::ShaderHandle shader = testDevice->createShader(desc,
        compiled.bytecode.data(), compiled.bytecode.size());
    REQUIRE(shader != nullptr);

    CHECK(shader->getDesc().shaderType == nvrhi::ShaderType::Pixel);
    CHECK_EQ(shader->getDesc().debugName, std::string("Solid color"));
    CHECK_EQ(shader->getDesc().entryName, std::string("main"));

    const void* bytecode = nullptr;
    size_t bytecodeSize = 0;
    shader->getBytecode(&bytecode, &bytecodeSize);
    REQUIRE(bytecode != nullptr);
    CHECK_EQ(bytecodeSize, compiled.bytecode.size());
    CHECK(std::memcmp(bytecode, compiled.bytecode.data(), bytecodeSize) == 0);
}

NVRHI_TEST_CASE("empty shader binaries are rejected")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    uint32_t dummy = 0;
    nvrhi::ShaderHandle shader = testDevice->createShader(
        nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Compute), &dummy, 0);
    CHECK(shader == nullptr);
}

NVRHI_TEST_CASE("shader libraries expose entries and bytecode")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    const nvrhi_test::CompiledShader compiled =
        nvrhi_test::compileShader(c_RayTracingLibrary, nullptr, "lib_6_6");
    REQUIRE(!compiled.bytecode.empty());

    nvrhi::ShaderLibraryHandle library = testDevice->createShaderLibrary(
        compiled.bytecode.data(), compiled.bytecode.size());
    REQUIRE(library != nullptr);

    const void* bytecode = nullptr;
    size_t bytecodeSize = 0;
    library->getBytecode(&bytecode, &bytecodeSize);
    REQUIRE(bytecode != nullptr);
    CHECK_EQ(bytecodeSize, compiled.bytecode.size());
    CHECK(std::memcmp(bytecode, compiled.bytecode.data(), bytecodeSize) == 0);

    nvrhi::ShaderHandle entry = library->getShader("raygen",
        nvrhi::ShaderType::RayGeneration);
    REQUIRE(entry != nullptr);
    CHECK_EQ(entry->getDesc().entryName, std::string("raygen"));
    CHECK(entry->getDesc().shaderType == nvrhi::ShaderType::RayGeneration);

    entry->getBytecode(&bytecode, &bytecodeSize);
    REQUIRE(bytecode != nullptr);
    CHECK_EQ(bytecodeSize, compiled.bytecode.size());
    CHECK(std::memcmp(bytecode, compiled.bytecode.data(), bytecodeSize) == 0);
}

NVRHI_TEST_CASE("graphics pipelines expose their desc and framebuffer info")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc desc;
    desc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::RenderTarget);

    nvrhi::TextureHandle color = testDevice->createTexture(desc);
    REQUIRE(color != nullptr);

    nvrhi::FramebufferHandle framebuffer = testDevice->createFramebuffer(
        nvrhi::FramebufferDesc().addColorAttachment(color));
    REQUIRE(framebuffer != nullptr);

    FullscreenPipeline bundle = createFullscreenPipeline(testDevice,
        framebuffer->getFramebufferInfo(), c_SolidColorPixelShader,
        nvrhi::DepthStencilState().disableDepthTest().disableDepthWrite());

    const nvrhi::GraphicsPipelineDesc& pipelineDesc = bundle.pipeline->getDesc();
    CHECK(pipelineDesc.VS.Get() == bundle.vertexShader.Get());
    CHECK(pipelineDesc.PS.Get() == bundle.pixelShader.Get());
    CHECK(pipelineDesc.primType == nvrhi::PrimitiveType::TriangleList);
    CHECK(bundle.pipeline->getFramebufferInfo() == framebuffer->getFramebufferInfo());

    const nvrhi::FramebufferDesc& framebufferDesc = framebuffer->getDesc();
    CHECK_EQ(framebufferDesc.colorAttachments.size(), size_t(1));
    CHECK(framebufferDesc.colorAttachments[0].texture == color.Get());
}

NVRHI_TEST_CASE("native pipeline state objects can be wrapped")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();
    nvrhi::d3d12::IDevice* testD3D12 = nvrhi_test::d3d12Device();
    ID3D12Device* native = nvrhi_test::nativeDevice();
    REQUIRE(testD3D12 != nullptr);
    REQUIRE(native != nullptr);

    nvrhi::TextureDesc colorDesc;
    colorDesc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::RenderTarget);
    nvrhi::TextureHandle color = testDevice->createTexture(colorDesc);
    REQUIRE(color != nullptr);

    nvrhi::FramebufferHandle framebuffer = testDevice->createFramebuffer(
        nvrhi::FramebufferDesc().addColorAttachment(color));
    REQUIRE(framebuffer != nullptr);

    nvrhi::ShaderHandle vertexShader = createShader(testDevice, nvrhi::ShaderType::Vertex,
        c_FullscreenVertexShader, "vs_6_0", "Native wrap vertex shader");
    nvrhi::ShaderHandle pixelShader = createShader(testDevice, nvrhi::ShaderType::Pixel,
        c_SolidColorPixelShader, "ps_6_0", "Native wrap pixel shader");
    REQUIRE(vertexShader != nullptr);
    REQUIRE(pixelShader != nullptr);

    nvrhi::GraphicsPipelineDesc desc;
    desc.setPrimType(nvrhi::PrimitiveType::TriangleList)
        .setVertexShader(vertexShader)
        .setPixelShader(pixelShader);
    desc.renderState.rasterState.setCullNone();
    desc.renderState.depthStencilState.disableDepthTest().disableDepthWrite();

    nvrhi::BindingLayoutVector layouts;
    nvrhi::d3d12::RootSignatureHandle rootSignature =
        testD3D12->buildRootSignature(layouts, false, false);
    REQUIRE(rootSignature != nullptr);

    const void* vertexBytecode = nullptr;
    size_t vertexBytecodeSize = 0;
    vertexShader->getBytecode(&vertexBytecode, &vertexBytecodeSize);
    const void* pixelBytecode = nullptr;
    size_t pixelBytecodeSize = 0;
    pixelShader->getBytecode(&pixelBytecode, &pixelBytecodeSize);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = static_cast<ID3D12RootSignature*>(
        rootSignature->getNativeObject(nvrhi::ObjectTypes::D3D12_RootSignature).pointer);
    psoDesc.VS = { vertexBytecode, vertexBytecodeSize };
    psoDesc.PS = { pixelBytecode, pixelBytecodeSize };

    D3D12_RENDER_TARGET_BLEND_DESC& blendTarget = psoDesc.BlendState.RenderTarget[0];
    blendTarget.BlendEnable = FALSE;
    blendTarget.SrcBlend = D3D12_BLEND_ONE;
    blendTarget.DestBlend = D3D12_BLEND_ZERO;
    blendTarget.BlendOp = D3D12_BLEND_OP_ADD;
    blendTarget.SrcBlendAlpha = D3D12_BLEND_ONE;
    blendTarget.DestBlendAlpha = D3D12_BLEND_ZERO;
    blendTarget.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blendTarget.LogicOp = D3D12_LOGIC_OP_NOOP;
    blendTarget.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.FrontCounterClockwise = FALSE;
    psoDesc.RasterizerState.DepthClipEnable = FALSE;
    psoDesc.RasterizerState.ConservativeRaster =
        D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.DepthStencilState.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
    psoDesc.DepthStencilState.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
    psoDesc.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    psoDesc.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    psoDesc.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    psoDesc.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    psoDesc.DepthStencilState.BackFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    psoDesc.DepthStencilState.BackFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    psoDesc.DepthStencilState.BackFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    psoDesc.DepthStencilState.BackFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;

    psoDesc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;

    Microsoft::WRL::ComPtr<ID3D12PipelineState> nativePipeline;
    REQUIRE(SUCCEEDED(native->CreateGraphicsPipelineState(&psoDesc,
        IID_PPV_ARGS(&nativePipeline))));

    nvrhi::GraphicsPipelineHandle wrapped = testD3D12->createHandleForNativeGraphicsPipeline(
        rootSignature.Get(), nativePipeline.Get(), desc,
        framebuffer->getFramebufferInfo());
    REQUIRE(wrapped != nullptr);
    CHECK(wrapped->getDesc().VS.Get() == vertexShader.Get());
    CHECK(wrapped->getFramebufferInfo() == framebuffer->getFramebufferInfo());
    CHECK(wrapped->getNativeObject(nvrhi::ObjectTypes::D3D12_PipelineState).pointer ==
        nativePipeline.Get());
}

NVRHI_TEST_CASE("full-screen draw writes the pixel shader color")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc desc;
    desc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::RenderTarget);

    nvrhi::TextureHandle color = testDevice->createTexture(desc);
    REQUIRE(color != nullptr);

    nvrhi::FramebufferHandle framebuffer = testDevice->createFramebuffer(
        nvrhi::FramebufferDesc().addColorAttachment(color));
    REQUIRE(framebuffer != nullptr);

    FullscreenPipeline bundle = createFullscreenPipeline(testDevice,
        framebuffer->getFramebufferInfo(), c_SolidColorPixelShader,
        nvrhi::DepthStencilState().disableDepthTest().disableDepthWrite());

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->setResourceStatesForFramebuffer(framebuffer);
    commandList->clearTextureFloat(color, nvrhi::AllSubresources,
        nvrhi::Color(0.f, 0.f, 0.f, 1.f));
    commandList->beginMarker("Full-screen draw");
    commandList->setGraphicsState(makeFullscreenState(bundle.pipeline, framebuffer, 8, 8));
    commandList->draw(nvrhi::DrawArguments().setVertexCount(3));
    commandList->endMarker();
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> pixels = nvrhi_test::readTexture2D(testDevice, color);
    checkAllPixels(pixels, 8, 8, 64, 128, 191, 255);
}

NVRHI_TEST_CASE("depth testing rejects farther fragments")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc colorDesc;
    colorDesc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::RenderTarget);
    nvrhi::TextureDesc depthDesc;
    depthDesc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::D32)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::DepthWrite);

    nvrhi::TextureHandle color = testDevice->createTexture(colorDesc);
    nvrhi::TextureHandle depth = testDevice->createTexture(depthDesc);
    REQUIRE(color != nullptr);
    REQUIRE(depth != nullptr);

    nvrhi::FramebufferHandle framebuffer = testDevice->createFramebuffer(
        nvrhi::FramebufferDesc().addColorAttachment(color).setDepthAttachment(depth));
    REQUIRE(framebuffer != nullptr);

    FullscreenPipeline green = createFullscreenPipeline(testDevice,
        framebuffer->getFramebufferInfo(), c_GreenPixelShader,
        nvrhi::DepthStencilState()
            .setDepthTestEnable(true)
            .setDepthWriteEnable(true)
            .setDepthFunc(nvrhi::ComparisonFunc::Less));

    FullscreenPipeline red = createFullscreenPipeline(testDevice,
        framebuffer->getFramebufferInfo(), c_RedPixelShader,
        nvrhi::DepthStencilState()
            .setDepthTestEnable(true)
            .setDepthWriteEnable(false)
            .setDepthFunc(nvrhi::ComparisonFunc::Greater));

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->clearTextureFloat(color, nvrhi::AllSubresources,
        nvrhi::Color(0.f, 0.f, 0.f, 1.f));
    commandList->clearDepthStencilTexture(depth, nvrhi::AllSubresources, true, 1.f, false, 0);
    commandList->setGraphicsState(makeFullscreenState(green.pipeline, framebuffer, 8, 8));
    commandList->draw(nvrhi::DrawArguments().setVertexCount(3));
    commandList->setGraphicsState(makeFullscreenState(red.pipeline, framebuffer, 8, 8));
    commandList->draw(nvrhi::DrawArguments().setVertexCount(3));
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> pixels = nvrhi_test::readTexture2D(testDevice, color);
    checkAllPixels(pixels, 8, 8, 0, 255, 0, 255);
}

NVRHI_TEST_CASE("indexed draws read the index buffer")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc colorDesc;
    colorDesc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::RenderTarget);
    nvrhi::TextureHandle color = testDevice->createTexture(colorDesc);
    REQUIRE(color != nullptr);

    nvrhi::FramebufferHandle framebuffer = testDevice->createFramebuffer(
        nvrhi::FramebufferDesc().addColorAttachment(color));
    REQUIRE(framebuffer != nullptr);

    FullscreenPipeline bundle = createFullscreenPipeline(testDevice,
        framebuffer->getFramebufferInfo(), c_GreenPixelShader,
        nvrhi::DepthStencilState().disableDepthTest().disableDepthWrite());

    const uint16_t indices[3] = { 0, 1, 2 };
    nvrhi::BufferHandle indexBuffer = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(indices))
        .setIsIndexBuffer(true)
        .setFormat(nvrhi::Format::R16_UINT)
        .setInitialState(nvrhi::ResourceStates::Common));
    REQUIRE(indexBuffer != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->writeBuffer(indexBuffer, indices, sizeof(indices));
    commandList->clearTextureFloat(color, nvrhi::AllSubresources,
        nvrhi::Color(0.f, 0.f, 0.f, 1.f));

    nvrhi::GraphicsState state = makeFullscreenState(bundle.pipeline, framebuffer, 8, 8);
    state.setIndexBuffer(nvrhi::IndexBufferBinding()
        .setBuffer(indexBuffer)
        .setFormat(nvrhi::Format::R16_UINT));
    commandList->setGraphicsState(state);
    commandList->drawIndexed(nvrhi::DrawArguments().setVertexCount(3));
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> pixels = nvrhi_test::readTexture2D(testDevice, color);
    checkAllPixels(pixels, 8, 8, 0, 255, 0, 255);
}

NVRHI_TEST_CASE("indirect draws read their arguments")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc colorDesc;
    colorDesc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::RenderTarget);
    nvrhi::TextureHandle color = testDevice->createTexture(colorDesc);
    REQUIRE(color != nullptr);

    nvrhi::FramebufferHandle framebuffer = testDevice->createFramebuffer(
        nvrhi::FramebufferDesc().addColorAttachment(color));
    REQUIRE(framebuffer != nullptr);

    FullscreenPipeline bundle = createFullscreenPipeline(testDevice,
        framebuffer->getFramebufferInfo(), c_SolidColorPixelShader,
        nvrhi::DepthStencilState().disableDepthTest().disableDepthWrite());

    nvrhi::DrawIndirectArguments arguments;
    arguments.vertexCount = 3;
    arguments.instanceCount = 1;

    nvrhi::BufferHandle indirectArgs = nvrhi_test::createUploadBuffer(testDevice,
        &arguments, sizeof(arguments), "Indirect draw arguments");

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->clearTextureFloat(color, nvrhi::AllSubresources,
        nvrhi::Color(0.f, 0.f, 0.f, 1.f));

    nvrhi::GraphicsState state = makeFullscreenState(bundle.pipeline, framebuffer, 8, 8);
    state.setIndirectParams(indirectArgs);
    commandList->setGraphicsState(state);
    commandList->drawIndirect(0, 1);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> pixels = nvrhi_test::readTexture2D(testDevice, color);
    checkAllPixels(pixels, 8, 8, 64, 128, 191, 255);
}

NVRHI_TEST_CASE("indexed indirect draws read their argument buffers")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc colorDesc;
    colorDesc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::RenderTarget);
    nvrhi::TextureHandle color = testDevice->createTexture(colorDesc);
    REQUIRE(color != nullptr);

    nvrhi::FramebufferHandle framebuffer = testDevice->createFramebuffer(
        nvrhi::FramebufferDesc().addColorAttachment(color));
    REQUIRE(framebuffer != nullptr);

    FullscreenPipeline bundle = createFullscreenPipeline(testDevice,
        framebuffer->getFramebufferInfo(), c_GreenPixelShader,
        nvrhi::DepthStencilState().disableDepthTest().disableDepthWrite());

    const uint16_t indices[3] = { 0, 1, 2 };
    nvrhi::BufferHandle indexBuffer = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(indices))
        .setIsIndexBuffer(true)
        .setFormat(nvrhi::Format::R16_UINT)
        .setInitialState(nvrhi::ResourceStates::Common));
    REQUIRE(indexBuffer != nullptr);

    nvrhi::DrawIndexedIndirectArguments arguments;
    arguments.indexCount = 3;
    arguments.instanceCount = 1;

    nvrhi::BufferHandle indirectArgs = nvrhi_test::createUploadBuffer(testDevice,
        &arguments, sizeof(arguments), "Indirect indexed draw arguments");

    const uint32_t drawCount = 1;
    nvrhi::BufferHandle countBuffer = nvrhi_test::createUploadBuffer(testDevice,
        &drawCount, sizeof(drawCount), "Indirect draw count");

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->writeBuffer(indexBuffer, indices, sizeof(indices));
    commandList->clearTextureFloat(color, nvrhi::AllSubresources,
        nvrhi::Color(0.f, 0.f, 0.f, 1.f));

    nvrhi::GraphicsState state = makeFullscreenState(bundle.pipeline, framebuffer, 8, 8);
    state.setIndexBuffer(nvrhi::IndexBufferBinding()
        .setBuffer(indexBuffer)
        .setFormat(nvrhi::Format::R16_UINT))
        .setIndirectParams(indirectArgs)
        .setIndirectCountBuffer(countBuffer);
    commandList->setGraphicsState(state);
    commandList->drawIndexedIndirectCount(0, 0, 1);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> pixels = nvrhi_test::readTexture2D(testDevice, color);
    checkAllPixels(pixels, 8, 8, 0, 255, 0, 255);

    nvrhi::CommandListHandle secondList = testDevice->createCommandList();
    REQUIRE(secondList != nullptr);
    secondList->open();
    secondList->clearTextureFloat(color, nvrhi::AllSubresources,
        nvrhi::Color(0.f, 0.f, 0.f, 1.f));
    state.setIndirectCountBuffer(nullptr);
    secondList->setGraphicsState(state);
    secondList->drawIndexedIndirect(0, 1);
    secondList->close();
    testDevice->executeCommandList(secondList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> secondPixels =
        nvrhi_test::readTexture2D(testDevice, color);
    checkAllPixels(secondPixels, 8, 8, 0, 255, 0, 255);
}

NVRHI_TEST_CASE("mesh shader pipelines render with direct and indirect dispatches")
{
    REQUIRE_DEVICE();

    nvrhi::IDevice* testDevice = nvrhi_test::device();
    SKIP_IF(!testDevice->queryFeatureSupport(nvrhi::Feature::Meshlets),
        "mesh shaders are not supported by this device");

    nvrhi::TextureDesc colorDesc;
    colorDesc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::RenderTarget);
    nvrhi::TextureHandle color = testDevice->createTexture(colorDesc);
    REQUIRE(color != nullptr);

    nvrhi::FramebufferHandle framebuffer = testDevice->createFramebuffer(
        nvrhi::FramebufferDesc().addColorAttachment(color));
    REQUIRE(framebuffer != nullptr);

    nvrhi::ShaderHandle meshShader = createShader(testDevice, nvrhi::ShaderType::Mesh,
        c_MeshShader, "ms_6_6", "Mesh shader");
    nvrhi::ShaderHandle pixelShader = createShader(testDevice, nvrhi::ShaderType::Pixel,
        c_SolidColorPixelShader, "ps_6_6", "Mesh pixel shader");
    REQUIRE(meshShader != nullptr);
    REQUIRE(pixelShader != nullptr);

    nvrhi::MeshletPipelineDesc pipelineDesc;
    pipelineDesc.setPrimType(nvrhi::PrimitiveType::TriangleList)
        .setMeshShader(meshShader)
        .setPixelShader(pixelShader);
    pipelineDesc.renderState.rasterState.setCullNone();
    pipelineDesc.renderState.depthStencilState.disableDepthTest().disableDepthWrite();

    nvrhi::MeshletPipelineHandle pipeline = testDevice->createMeshletPipeline(
        pipelineDesc, framebuffer->getFramebufferInfo());
    REQUIRE(pipeline != nullptr);
    CHECK(pipeline->getDesc().MS.Get() == meshShader.Get());
    CHECK(pipeline->getFramebufferInfo() == framebuffer->getFramebufferInfo());

    const auto makeState = [&]()
    {
        nvrhi::MeshletState state;
        state.setPipeline(pipeline)
            .setFramebuffer(framebuffer)
            .setViewport(nvrhi::ViewportState().addViewportAndScissorRect(
                nvrhi::Viewport(8.f, 8.f)));
        return state;
    };

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->clearTextureFloat(color, nvrhi::AllSubresources,
        nvrhi::Color(0.f, 0.f, 0.f, 1.f));
    commandList->setMeshletState(makeState());
    commandList->dispatchMesh(1);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());
    checkAllPixels(nvrhi_test::readTexture2D(testDevice, color), 8, 8, 64, 128, 191, 255);

    nvrhi::DispatchIndirectArguments dispatchArgs;
    dispatchArgs.setGroups3D(1, 1, 1);
    nvrhi::BufferHandle indirectArgs = nvrhi_test::createUploadBuffer(testDevice,
        &dispatchArgs, sizeof(dispatchArgs), "Mesh dispatch arguments");

    nvrhi::MeshletState indirectState = makeState();
    indirectState.setIndirectParams(indirectArgs);

    nvrhi::CommandListHandle indirectList = testDevice->createCommandList();
    REQUIRE(indirectList != nullptr);
    indirectList->open();
    indirectList->clearTextureFloat(color, nvrhi::AllSubresources,
        nvrhi::Color(0.f, 0.f, 0.f, 1.f));
    indirectList->setMeshletState(indirectState);
    indirectList->dispatchMeshIndirect(0, 1);
    indirectList->close();
    testDevice->executeCommandList(indirectList);
    REQUIRE(testDevice->waitForIdle());
    checkAllPixels(nvrhi_test::readTexture2D(testDevice, color), 8, 8, 64, 128, 191, 255);

    const uint32_t drawCount = 1;
    nvrhi::BufferHandle countBuffer = nvrhi_test::createUploadBuffer(testDevice,
        &drawCount, sizeof(drawCount), "Mesh dispatch count");

    indirectState.setIndirectCountBuffer(countBuffer);

    nvrhi::CommandListHandle countList = testDevice->createCommandList();
    REQUIRE(countList != nullptr);
    countList->open();
    countList->clearTextureFloat(color, nvrhi::AllSubresources,
        nvrhi::Color(0.f, 0.f, 0.f, 1.f));
    countList->setMeshletState(indirectState);
    countList->dispatchMeshIndirectCount(0, 0, 1);
    countList->close();
    testDevice->executeCommandList(countList);
    REQUIRE(testDevice->waitForIdle());
    checkAllPixels(nvrhi_test::readTexture2D(testDevice, color), 8, 8, 64, 128, 191, 255);
}

NVRHI_TEST_CASE("clearTextureFloat fills a render target")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc desc;
    desc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::RenderTarget);
    nvrhi::TextureHandle color = testDevice->createTexture(desc);
    REQUIRE(color != nullptr);

    nvrhi::FramebufferHandle framebuffer = testDevice->createFramebuffer(
        nvrhi::FramebufferDesc().addColorAttachment(color));
    REQUIRE(framebuffer != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->clearTextureFloat(color, nvrhi::AllSubresources,
        nvrhi::Color(0.25f, 0.5f, 0.75f, 1.f));
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> pixels = nvrhi_test::readTexture2D(testDevice, color);
    checkAllPixels(pixels, 8, 8, 64, 128, 191, 255);

    nvrhi::CommandListHandle secondList = testDevice->createCommandList();
    REQUIRE(secondList != nullptr);
    secondList->open();
    nvrhi::utils::ClearColorAttachment(secondList, framebuffer, 0,
        nvrhi::Color(0.f, 1.f, 0.f, 1.f));
    secondList->close();
    testDevice->executeCommandList(secondList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> green = nvrhi_test::readTexture2D(testDevice, color);
    checkAllPixels(green, 8, 8, 0, 255, 0, 255);
}

NVRHI_TEST_CASE("clearTextureUInt fills a UAV texture")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc desc;
    desc.setWidth(4).setHeight(4).setFormat(nvrhi::Format::R32_UINT)
        .setIsUAV(true).setInitialState(nvrhi::ResourceStates::UnorderedAccess);
    nvrhi::TextureHandle texture = testDevice->createTexture(desc);
    REQUIRE(texture != nullptr);

    const uint32_t clearValue = 0xDEADBEEFu;
    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->setEnableUavBarriersForTexture(texture, false);
    commandList->clearTextureUInt(texture, nvrhi::AllSubresources, clearValue);
    commandList->setEnableUavBarriersForTexture(texture, true);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> pixels = nvrhi_test::readTexture2D(testDevice, texture);
    REQUIRE(pixels.size() == size_t(4 * 4 * 4));
    for (size_t offset = 0; offset < pixels.size(); offset += 4)
    {
        uint32_t value = 0;
        std::memcpy(&value, pixels.data() + offset, sizeof(value));
        CHECK_EQ(value, clearValue);
    }
}

NVRHI_TEST_CASE("writeTexture uploads pixel data")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc desc;
    desc.setWidth(8).setHeight(4).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setInitialState(nvrhi::ResourceStates::CopyDest);
    nvrhi::TextureHandle texture = testDevice->createTexture(desc);
    REQUIRE(texture != nullptr);

    std::vector<uint8_t> source(8 * 4 * 4);
    for (uint32_t y = 0; y < 4; ++y)
    {
        for (uint32_t x = 0; x < 8; ++x)
        {
            const size_t offset = (size_t(y) * 8 + x) * 4;
            source[offset + 0] = uint8_t(x * 16);
            source[offset + 1] = uint8_t(y * 32);
            source[offset + 2] = uint8_t(x + y);
            source[offset + 3] = 255;
        }
    }

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->writeTexture(texture, 0, 0, source.data(), 8 * 4);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> pixels = nvrhi_test::readTexture2D(testDevice, texture);
    CHECK(pixels == source);

    nvrhi::TextureHandle sliceTexture = testDevice->createTexture(desc);
    REQUIRE(sliceTexture != nullptr);

    const nvrhi::TextureSlice fullSlice = nvrhi::TextureSlice().resolve(desc);
    nvrhi::CommandListHandle sliceList = testDevice->createCommandList();
    REQUIRE(sliceList != nullptr);
    sliceList->open();
    sliceList->writeTexture(sliceTexture, fullSlice, source.data(), 8 * 4);
    sliceList->close();
    testDevice->executeCommandList(sliceList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> slicePixels =
        nvrhi_test::readTexture2D(testDevice, sliceTexture);
    CHECK(slicePixels == source);
}

NVRHI_TEST_CASE("staging textures upload pixels through copyTexture")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc desc;
    desc.setWidth(8).setHeight(4).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setInitialState(nvrhi::ResourceStates::CopyDest);

    nvrhi::TextureHandle texture = testDevice->createTexture(desc);
    REQUIRE(texture != nullptr);

    nvrhi::StagingTextureHandle staging =
        testDevice->createStagingTexture(desc, nvrhi::CpuAccessMode::Write);
    REQUIRE(staging != nullptr);

    const nvrhi::TextureSlice slice = nvrhi::TextureSlice().resolve(desc);
    size_t rowPitch = 0;
    uint8_t* mapped = static_cast<uint8_t*>(testDevice->mapStagingTexture(
        staging, slice, nvrhi::CpuAccessMode::Write, &rowPitch));
    REQUIRE(mapped != nullptr);

    std::vector<uint8_t> expected(8 * 4 * 4);
    for (uint32_t y = 0; y < 4; ++y)
    {
        for (uint32_t x = 0; x < 8; ++x)
        {
            const size_t offset = (size_t(y) * 8 + x) * 4;
            expected[offset + 0] = uint8_t(x * 16);
            expected[offset + 1] = uint8_t(y * 32);
            expected[offset + 2] = uint8_t(x + y);
            expected[offset + 3] = 255;
        }
        std::memcpy(mapped + size_t(y) * rowPitch, expected.data() + size_t(y) * 8 * 4, 8 * 4);
    }
    testDevice->unmapStagingTexture(staging);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->copyTexture(texture, slice, staging, slice);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> pixels = nvrhi_test::readTexture2D(testDevice, texture);
    CHECK(pixels == expected);
}

NVRHI_TEST_CASE("copyTexture round-trips through a second texture")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc desc;
    desc.setWidth(8).setHeight(4).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setInitialState(nvrhi::ResourceStates::CopyDest);

    nvrhi::TextureHandle source = testDevice->createTexture(desc);
    nvrhi::TextureHandle destination = testDevice->createTexture(desc);
    REQUIRE(source != nullptr);
    REQUIRE(destination != nullptr);

    std::vector<uint8_t> pixels(8 * 4 * 4);
    for (size_t index = 0; index < pixels.size(); ++index)
        pixels[index] = uint8_t(index);

    const nvrhi::TextureSlice slice = nvrhi::TextureSlice().resolve(desc);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->writeTexture(source, 0, 0, pixels.data(), 8 * 4);
    commandList->copyTexture(destination, slice, source, slice);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> copied = nvrhi_test::readTexture2D(testDevice, destination);
    CHECK(copied == pixels);
}

NVRHI_TEST_CASE("resolveTexture collapses multisampled color")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc multisampledDesc;
    multisampledDesc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setDimension(nvrhi::TextureDimension::Texture2DMS)
        .setSampleCount(4).setIsRenderTarget(true)
        .setInitialState(nvrhi::ResourceStates::RenderTarget);

    nvrhi::TextureDesc resolvedDesc;
    resolvedDesc.setWidth(8).setHeight(8).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsRenderTarget(true).setInitialState(nvrhi::ResourceStates::RenderTarget);

    nvrhi::TextureHandle multisampled = testDevice->createTexture(multisampledDesc);
    nvrhi::TextureHandle resolved = testDevice->createTexture(resolvedDesc);
    REQUIRE(multisampled != nullptr);
    REQUIRE(resolved != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->clearTextureFloat(multisampled, nvrhi::AllSubresources,
        nvrhi::Color(0.f, 1.f, 0.f, 1.f));
    commandList->resolveTexture(resolved, nvrhi::AllSubresources, multisampled,
        nvrhi::AllSubresources);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> pixels = nvrhi_test::readTexture2D(testDevice, resolved);
    checkAllPixels(pixels, 8, 8, 0, 255, 0, 255);
}

NVRHI_TEST_CASE("compute dispatch writes a structured buffer")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::ShaderHandle computeShader = createShader(testDevice, nvrhi::ShaderType::Compute,
        c_ClearComputeShader, "cs_6_0", "Clear compute shader");
    REQUIRE(computeShader != nullptr);

    nvrhi::BufferDesc outputDesc;
    outputDesc.byteSize = 32 * 4;
    outputDesc.structStride = 4;
    outputDesc.canHaveUAVs = true;
    outputDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;

    nvrhi::BufferHandle output = testDevice->createBuffer(outputDesc);
    REQUIRE(output != nullptr);

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.setVisibility(nvrhi::ShaderType::Compute)
        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0));
    nvrhi::BindingLayoutHandle layout = testDevice->createBindingLayout(layoutDesc);
    REQUIRE(layout != nullptr);

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, output));
    nvrhi::BindingSetHandle bindingSet = testDevice->createBindingSet(setDesc, layout);
    REQUIRE(bindingSet != nullptr);

    nvrhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.setComputeShader(computeShader).addBindingLayout(layout);
    nvrhi::ComputePipelineHandle pipeline = testDevice->createComputePipeline(pipelineDesc);
    REQUIRE(pipeline != nullptr);
    CHECK(pipeline->getDesc().CS.Get() == computeShader.Get());
    CHECK_EQ(pipeline->getDesc().bindingLayouts.size(), size_t(1));

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->setEnableUavBarriersForBuffer(output, false);
    commandList->setResourceStatesForBindingSet(bindingSet);
    commandList->setComputeState(nvrhi::ComputeState()
        .setPipeline(pipeline)
        .addBindingSet(bindingSet));
    commandList->dispatch(1);
    commandList->setEnableUavBarriersForBuffer(output, true);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> data = nvrhi_test::readBuffer(testDevice, output, 32 * 4);
    for (uint32_t index = 0; index < 32; ++index)
    {
        uint32_t value = 0;
        std::memcpy(&value, data.data() + size_t(index) * 4, sizeof(value));
        CHECK_EQ(value, index * 3 + 1);
    }
}

NVRHI_TEST_CASE("compute push constants reach the shader")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::ShaderHandle computeShader = createShader(testDevice, nvrhi::ShaderType::Compute,
        c_PushConstantComputeShader, "cs_6_0", "Push constant compute shader");
    REQUIRE(computeShader != nullptr);

    nvrhi::BufferDesc outputDesc;
    outputDesc.byteSize = 32 * 4;
    outputDesc.structStride = 4;
    outputDesc.canHaveUAVs = true;
    outputDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;

    nvrhi::BufferHandle output = testDevice->createBuffer(outputDesc);
    REQUIRE(output != nullptr);

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.setVisibility(nvrhi::ShaderType::Compute)
        .addItem(nvrhi::BindingLayoutItem::PushConstants(0, sizeof(uint32_t)))
        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0));
    nvrhi::BindingLayoutHandle layout = testDevice->createBindingLayout(layoutDesc);
    REQUIRE(layout != nullptr);

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, output));
    nvrhi::BindingSetHandle bindingSet = testDevice->createBindingSet(setDesc, layout);
    REQUIRE(bindingSet != nullptr);

    nvrhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.setComputeShader(computeShader).addBindingLayout(layout);
    nvrhi::ComputePipelineHandle pipeline = testDevice->createComputePipeline(pipelineDesc);
    REQUIRE(pipeline != nullptr);

    const uint32_t multiplier = 7;
    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->setComputeState(nvrhi::ComputeState()
        .setPipeline(pipeline)
        .addBindingSet(bindingSet));
    commandList->setPushConstants(&multiplier, sizeof(multiplier));
    commandList->dispatch(1);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> data = nvrhi_test::readBuffer(testDevice, output, 32 * 4);
    for (uint32_t index = 0; index < 32; ++index)
    {
        uint32_t value = 0;
        std::memcpy(&value, data.data() + size_t(index) * 4, sizeof(value));
        CHECK_EQ(value, index * multiplier);
    }
}

NVRHI_TEST_CASE("clearBufferUInt fills a UAV buffer")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BufferHandle buffer = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(256)
        .setCanHaveUAVs(true)
        .setInitialState(nvrhi::ResourceStates::UnorderedAccess));
    REQUIRE(buffer != nullptr);

    const uint32_t clearValue = 0x11223344u;
    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->clearBufferUInt(buffer, clearValue);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> data = nvrhi_test::readBuffer(testDevice, buffer, 256);
    for (size_t offset = 0; offset < data.size(); offset += 4)
    {
        uint32_t value = 0;
        std::memcpy(&value, data.data() + offset, sizeof(value));
        CHECK_EQ(value, clearValue);
    }
}

NVRHI_TEST_CASE("copyBuffer copies between buffers")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    std::vector<uint8_t> source(64);
    for (size_t index = 0; index < source.size(); ++index)
        source[index] = uint8_t(0xA0 + index);

    nvrhi::BufferHandle upload = nvrhi_test::createUploadBuffer(testDevice,
        source.data(), source.size(), "Copy source");

    nvrhi::BufferHandle deviceBuffer = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(source.size())
        .setInitialState(nvrhi::ResourceStates::Common));
    REQUIRE(deviceBuffer != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->copyBuffer(deviceBuffer, 0, upload, 0, source.size());
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> copied = nvrhi_test::readBuffer(testDevice,
        deviceBuffer, source.size());
    CHECK(copied == source);
}

NVRHI_TEST_CASE("writeHeap uploads into placed heap memory")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::HeapHandle heap = testDevice->createHeap(nvrhi::HeapDesc()
        .setCapacity(65536)
        .setType(nvrhi::HeapType::DeviceLocal)
        .setDebugName("Heap upload target"));
    REQUIRE(heap != nullptr);

    const uint8_t data[16] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16 };

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->writeHeap(heap, 0, data, sizeof(data));
    commandList->close();
    testDevice->executeCommandList(commandList);
    CHECK(testDevice->waitForIdle());
}

NVRHI_TEST_CASE("acceleration structures build and expose device addresses")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    SKIP_IF(!testDevice->queryFeatureSupport(nvrhi::Feature::RayTracingAccelStruct),
        "ray tracing is not supported by this device");

    const float vertices[9] = {
        0.f, 0.f, 0.f,
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f,
    };

    nvrhi::BufferHandle vertexBuffer = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(vertices))
        .setIsAccelStructBuildInput(true)
        .setInitialState(nvrhi::ResourceStates::AccelStructBuildInput));
    REQUIRE(vertexBuffer != nullptr);

    const uint16_t indices[3] = { 0, 1, 2 };
    nvrhi::BufferHandle indexBuffer = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(indices))
        .setIsAccelStructBuildInput(true)
        .setInitialState(nvrhi::ResourceStates::AccelStructBuildInput));
    REQUIRE(indexBuffer != nullptr);

    nvrhi::rt::GeometryTriangles triangles;
    triangles.setVertexBuffer(vertexBuffer)
        .setVertexFormat(nvrhi::Format::RGB32_FLOAT)
        .setVertexCount(3)
        .setVertexStride(12)
        .setIndexBuffer(indexBuffer)
        .setIndexFormat(nvrhi::Format::R16_UINT)
        .setIndexCount(3);

    const nvrhi::rt::GeometryDesc geometry =
        nvrhi::rt::GeometryDesc().setTriangles(triangles);

    nvrhi::rt::AccelStructDesc blasDesc;
    blasDesc.addBottomLevelGeometry(geometry)
        .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);

    nvrhi::rt::AccelStructHandle blas = testDevice->createAccelStruct(blasDesc);
    REQUIRE(blas != nullptr);
    CHECK(!blas->getDesc().isTopLevel);
    CHECK_EQ(blas->getDesc().bottomLevelGeometries.size(), size_t(1));

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->writeBuffer(vertexBuffer, vertices, sizeof(vertices));
    commandList->writeBuffer(indexBuffer, indices, sizeof(indices));
    commandList->setAccelStructState(blas, nvrhi::ResourceStates::AccelStructWrite);
    commandList->buildBottomLevelAccelStruct(blas, &geometry, 1);
    commandList->compactBottomLevelAccelStructs();
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    CHECK(blas->getDeviceAddress() != 0);
    CHECK(!blas->isCompacted());

    nvrhi::rt::AccelStructHandle cloned = testDevice->createAccelStruct(blasDesc);
    REQUIRE(cloned != nullptr);

    commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->copyRaytracingAccelerationStructure(cloned, blas);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    CHECK(cloned->getDeviceAddress() != 0);

    nvrhi::rt::AccelStructDesc tlasDesc;
    tlasDesc.setTopLevelMaxInstances(1)
        .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);

    nvrhi::rt::AccelStructPrebuildInfo prebuildInfo = {};
    if (testDevice->queryTopLevelAccelStructPrebuildInfo(tlasDesc, 1, prebuildInfo))
    {
        CHECK(prebuildInfo.resultBytes > 0);
        CHECK(prebuildInfo.scratchBytes > 0);
    }

    nvrhi::rt::AccelStructHandle tlas = testDevice->createAccelStruct(tlasDesc);
    REQUIRE(tlas != nullptr);
    CHECK(tlas->getDesc().isTopLevel);
    CHECK_EQ(tlas->getDesc().topLevelMaxInstances, size_t(1));

    nvrhi::rt::InstanceDesc instance;
    instance.setInstanceMask(0xFF).setBLAS(blas);

    commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->buildTopLevelAccelStruct(tlas, &instance, 1);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    CHECK(tlas->getDeviceAddress() != 0);
    CHECK(!tlas->isCompacted());

    nvrhi::rt::InstanceDesc instanceFromBuffer;
    instanceFromBuffer.setInstanceMask(0xFF);
    instanceFromBuffer.blasDeviceAddress = blas->getDeviceAddress();

    nvrhi::BufferHandle instanceBuffer = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(sizeof(instanceFromBuffer))
        .setIsAccelStructBuildInput(true)
        .setInitialState(nvrhi::ResourceStates::AccelStructBuildInput));
    REQUIRE(instanceBuffer != nullptr);

    nvrhi::rt::AccelStructHandle tlasFromBuffer = testDevice->createAccelStruct(tlasDesc);
    REQUIRE(tlasFromBuffer != nullptr);

    commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->writeBuffer(instanceBuffer, &instanceFromBuffer, sizeof(instanceFromBuffer));
    commandList->buildTopLevelAccelStructFromBuffer(tlasFromBuffer, instanceBuffer, 0, 1);
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    CHECK(tlasFromBuffer->getDeviceAddress() != 0);

    nvrhi::rt::AccelStructDesc virtualDesc;
    virtualDesc.setTopLevelMaxInstances(1)
        .setIsVirtual(true)
        .setBuildFlags(nvrhi::rt::AccelStructBuildFlags::PreferFastTrace);

    nvrhi::rt::AccelStructHandle virtualAs = testDevice->createAccelStruct(virtualDesc);
    REQUIRE(virtualAs != nullptr);

    const nvrhi::MemoryRequirements asRequirements =
        testDevice->getAccelStructMemoryRequirements(virtualAs);
    REQUIRE(asRequirements.size > 0);

    nvrhi::HeapHandle asHeap = testDevice->createHeap(nvrhi::HeapDesc()
        .setCapacity(nvrhi::align(asRequirements.size, 65536ull))
        .setType(nvrhi::HeapType::DeviceLocal));
    REQUIRE(asHeap != nullptr);

    CHECK(testDevice->bindAccelStructMemory(virtualAs, asHeap, 0));
    CHECK(!testDevice->bindAccelStructMemory(virtualAs, asHeap, 0));
    CHECK(!testDevice->bindAccelStructMemory(tlas, asHeap, 0));

    nvrhi::MemoryRequirements boundRequirements = {};
    CHECK(virtualAs->queryMemoryRequirements(boundRequirements));
    CHECK_EQ(boundRequirements.size, asRequirements.size);
    CHECK_EQ(boundRequirements.alignment, asRequirements.alignment);
}

NVRHI_TEST_CASE("ray tracing pipelines dispatch through shader tables")
{
    REQUIRE_DEVICE();

    nvrhi::IDevice* testDevice = nvrhi_test::device();
    SKIP_IF(!testDevice->queryFeatureSupport(nvrhi::Feature::RayTracingPipeline),
        "ray tracing pipelines are not supported by this device");

    const nvrhi_test::CompiledShader libraryBytes =
        nvrhi_test::compileShader(c_RayTracingLibrary, nullptr, "lib_6_6");
    REQUIRE(!libraryBytes.bytecode.empty());

    const auto makeShader = [&](const char* entry, nvrhi::ShaderType type)
    {
        return testDevice->createShader(
            nvrhi::ShaderDesc()
                .setShaderType(type)
                .setEntryName(entry)
                .setDebugName(entry),
            libraryBytes.bytecode.data(), libraryBytes.bytecode.size());
    };

    nvrhi::ShaderHandle rayGeneration =
        makeShader("raygen", nvrhi::ShaderType::RayGeneration);
    nvrhi::ShaderHandle miss = makeShader("miss", nvrhi::ShaderType::Miss);
    nvrhi::ShaderHandle closestHit =
        makeShader("closest", nvrhi::ShaderType::ClosestHit);
    nvrhi::ShaderHandle callable = makeShader("callable", nvrhi::ShaderType::Callable);
    REQUIRE(rayGeneration != nullptr);
    REQUIRE(miss != nullptr);
    REQUIRE(closestHit != nullptr);
    REQUIRE(callable != nullptr);

    nvrhi::BufferDesc outputDesc;
    outputDesc.byteSize = 16;
    outputDesc.structStride = 4;
    outputDesc.canHaveUAVs = true;
    outputDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    nvrhi::BufferHandle output = testDevice->createBuffer(outputDesc);
    REQUIRE(output != nullptr);

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.setVisibility(nvrhi::ShaderType::All)
        .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0));
    nvrhi::BindingLayoutHandle layout = testDevice->createBindingLayout(layoutDesc);
    REQUIRE(layout != nullptr);

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, output));
    nvrhi::BindingSetHandle bindingSet = testDevice->createBindingSet(setDesc, layout);
    REQUIRE(bindingSet != nullptr);

    nvrhi::rt::PipelineDesc pipelineDesc;
    pipelineDesc.setMaxPayloadSize(4)
        .setMaxAttributeSize(8)
        .setMaxRecursionDepth(1)
        .addBindingLayout(layout)
        .addShader(nvrhi::rt::PipelineShaderDesc()
            .setExportName("raygen").setShader(rayGeneration))
        .addShader(nvrhi::rt::PipelineShaderDesc()
            .setExportName("miss").setShader(miss))
        .addShader(nvrhi::rt::PipelineShaderDesc()
            .setExportName("callable").setShader(callable))
        .addHitGroup(nvrhi::rt::PipelineHitGroupDesc()
            .setExportName("closest").setClosestHitShader(closestHit));

    nvrhi::rt::PipelineHandle pipeline =
        testDevice->createRayTracingPipeline(pipelineDesc);
    REQUIRE(pipeline != nullptr);
    CHECK_EQ(pipeline->getDesc().shaders.size(), size_t(3));
    CHECK_EQ(pipeline->getDesc().hitGroups.size(), size_t(1));

    nvrhi::rt::ShaderTableHandle uncached = pipeline->createShaderTable();
    REQUIRE(uncached != nullptr);
    CHECK(!uncached->getDesc().isCached);
    // The ray generation entry is reserved from the start.
    CHECK_EQ(uncached->getNumEntries(), 1u);
    CHECK(uncached->getPipeline() == pipeline.Get());

    nvrhi::rt::ShaderTableDesc cachedDesc;
    cachedDesc.enableCaching(16);
    nvrhi::rt::ShaderTableHandle table = pipeline->createShaderTable(cachedDesc);
    REQUIRE(table != nullptr);
    CHECK(table->getDesc().isCached);
    CHECK_EQ(table->getDesc().maxEntries, 16u);
    CHECK(table->getPipeline() == pipeline.Get());

    table->setRayGenerationShader("raygen");
    CHECK_EQ(table->getNumEntries(), 1u);
    CHECK_EQ(table->addMissShader("miss"), 0);
    CHECK_EQ(table->addHitGroup("closest"), 0);
    CHECK_EQ(table->addCallableShader("callable"), 0);
    CHECK_EQ(table->getNumEntries(), 4u);

    table->clearCallableShaders();
    CHECK_EQ(table->getNumEntries(), 3u);
    table->clearHitShaders();
    CHECK_EQ(table->getNumEntries(), 2u);
    table->clearMissShaders();
    CHECK_EQ(table->getNumEntries(), 1u);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->setRayTracingState(nvrhi::rt::State()
        .setShaderTable(table)
        .addBindingSet(bindingSet));
    commandList->dispatchRays(nvrhi::rt::DispatchRaysArguments().setDimensions(1, 1, 1));
    commandList->close();
    testDevice->executeCommandList(commandList);
    REQUIRE(testDevice->waitForIdle());

    const std::vector<uint8_t> data = nvrhi_test::readBuffer(testDevice, output, 16);
    uint32_t value = 0;
    std::memcpy(&value, data.data(), sizeof(value));
    CHECK_EQ(value, 42u);
}
