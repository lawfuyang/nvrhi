#include "test-harness.h"

#include <nvrhi/common/misc.h>
#include <nvrhi/utils.h>

#include <dxgi.h>
#include <wrl/client.h>

#include <cstring>
#include <utility>

namespace
{
    nvrhi::TextureDesc makeColorTextureDesc(uint32_t width, uint32_t height)
    {
        return nvrhi::TextureDesc()
            .setWidth(width)
            .setHeight(height)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setIsRenderTarget(true)
            .setInitialState(nvrhi::ResourceStates::RenderTarget);
    }

    nvrhi::TextureDesc makeDepthTextureDesc(uint32_t width, uint32_t height)
    {
        return nvrhi::TextureDesc()
            .setWidth(width)
            .setHeight(height)
            .setFormat(nvrhi::Format::D32)
            .setIsRenderTarget(true)
            .setInitialState(nvrhi::ResourceStates::DepthWrite);
    }
}

NVRHI_TEST_CASE("device reports D3D12 and native handles")
{
    REQUIRE_DEVICE();

    nvrhi::IDevice* testDevice = nvrhi_test::device();
    REQUIRE(testDevice != nullptr);

    CHECK(testDevice->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12);
    CHECK(testDevice->getMessageCallback() == nvrhi_test::messageCallback());
    CHECK(testDevice->getNativeObject(nvrhi::ObjectTypes::D3D12_Device).pointer ==
        nvrhi_test::nativeDevice());
    CHECK(testDevice->getNativeObject(
        nvrhi::ObjectTypes::Nvrhi_D3D12_Device).pointer != nullptr);
    CHECK(testDevice->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue,
        nvrhi::CommandQueue::Graphics).pointer != nullptr);
}

NVRHI_TEST_CASE("feature queries can be issued for every feature")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    for (uint32_t index = 0; index <= uint32_t(nvrhi::Feature::EnhancedBarriers); ++index)
    {
        const nvrhi::Feature feature = nvrhi::Feature(index);
        const bool supported = testDevice->queryFeatureSupport(feature);
        if (feature == nvrhi::Feature::ComputeQueue)
            CHECK(supported);
        if (feature == nvrhi::Feature::CopyQueue)
            CHECK(supported);
        if (feature == nvrhi::Feature::DeferredCommandLists)
            CHECK(supported);
        if (feature == nvrhi::Feature::VirtualResources)
            CHECK(supported);
    }

    nvrhi::VariableRateShadingFeatureInfo variableRateShading = {};
    testDevice->queryFeatureSupport(nvrhi::Feature::VariableRateShading,
        &variableRateShading, sizeof(variableRateShading));

    nvrhi::WaveLaneCountMinMaxFeatureInfo waveLanes = {};
    if (testDevice->queryFeatureSupport(nvrhi::Feature::WaveLaneCountMinMax,
        &waveLanes, sizeof(waveLanes)))
    {
        CHECK(waveLanes.minWaveLaneCount > 0);
        CHECK(waveLanes.maxWaveLaneCount >= waveLanes.minWaveLaneCount);
    }

    nvrhi::RayTracingClustersFeatureInfo clusters = { nvrhi::ClusterApi::None };
    if (testDevice->queryFeatureSupport(nvrhi::Feature::RayTracingClusters,
        &clusters, sizeof(clusters)))
    {
        CHECK(clusters.api != nvrhi::ClusterApi::None);
    }
}

NVRHI_TEST_CASE("format support reflects common usages")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    const nvrhi::FormatSupport color =
        testDevice->queryFormatSupport(nvrhi::Format::RGBA8_UNORM);
    CHECK((color & nvrhi::FormatSupport::Texture) == nvrhi::FormatSupport::Texture);
    CHECK((color & nvrhi::FormatSupport::RenderTarget) == nvrhi::FormatSupport::RenderTarget);
    CHECK((color & nvrhi::FormatSupport::Blendable) == nvrhi::FormatSupport::Blendable);
    CHECK((color & nvrhi::FormatSupport::ShaderSample) == nvrhi::FormatSupport::ShaderSample);

    const nvrhi::FormatSupport depth =
        testDevice->queryFormatSupport(nvrhi::Format::D32);
    CHECK((depth & nvrhi::FormatSupport::DepthStencil) == nvrhi::FormatSupport::DepthStencil);
    CHECK((depth & nvrhi::FormatSupport::Texture) == nvrhi::FormatSupport::Texture);

    const nvrhi::FormatSupport compressed =
        testDevice->queryFormatSupport(nvrhi::Format::BC1_UNORM);
    CHECK((compressed & nvrhi::FormatSupport::Texture) == nvrhi::FormatSupport::Texture);
    CHECK((compressed & nvrhi::FormatSupport::ShaderSample) == nvrhi::FormatSupport::ShaderSample);

    const nvrhi::FormatSupport integer =
        testDevice->queryFormatSupport(nvrhi::Format::R32_UINT);
    CHECK((integer & nvrhi::FormatSupport::Buffer) == nvrhi::FormatSupport::Buffer);
    CHECK((integer & nvrhi::FormatSupport::Texture) == nvrhi::FormatSupport::Texture);
}

NVRHI_TEST_CASE("descriptor heaps allocate and release descriptors")
{
    REQUIRE_DEVICE();
    nvrhi::d3d12::IDevice* testDevice = nvrhi_test::d3d12Device();
    REQUIRE(testDevice != nullptr);

    for (nvrhi::d3d12::DescriptorHeapType heapType : {
        nvrhi::d3d12::DescriptorHeapType::RenderTargetView,
        nvrhi::d3d12::DescriptorHeapType::DepthStencilView,
        nvrhi::d3d12::DescriptorHeapType::ShaderResourceView,
        nvrhi::d3d12::DescriptorHeapType::Sampler })
    {
        nvrhi::d3d12::IDescriptorHeap* heap = testDevice->getDescriptorHeap(heapType);
        REQUIRE(heap != nullptr);
        CHECK(heap->getHeap() != nullptr);

        const bool shaderVisible = heapType == nvrhi::d3d12::DescriptorHeapType::ShaderResourceView ||
            heapType == nvrhi::d3d12::DescriptorHeapType::Sampler;
        if (shaderVisible)
            CHECK(heap->getShaderVisibleHeap() != nullptr);
        else
            CHECK(heap->getShaderVisibleHeap() == nullptr);

        const nvrhi::d3d12::DescriptorIndex base = heap->allocateDescriptors(4);
        const nvrhi::d3d12::DescriptorIndex single = heap->allocateDescriptor();
        CHECK_NE(single, base);

        const D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->getCpuHandle(base);
        CHECK(cpu.ptr != 0);
        const D3D12_CPU_DESCRIPTOR_HANDLE visibleCpu = heap->getCpuHandleShaderVisible(base);
        const D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->getGpuHandle(base);
        if (shaderVisible)
        {
            CHECK(visibleCpu.ptr != 0);
            CHECK(gpu.ptr != 0);
        }

        heap->releaseDescriptor(single);
        heap->releaseDescriptors(base, 4);
    }
}

NVRHI_TEST_CASE("buffer creation reports its description and native resource")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BufferDesc desc;
    desc.byteSize = 1024;
    desc.debugName = "Test buffer";
    desc.isVertexBuffer = true;
    desc.isIndexBuffer = true;
    desc.canHaveRawViews = true;

    nvrhi::BufferHandle buffer = testDevice->createBuffer(desc);
    REQUIRE(buffer != nullptr);

    const nvrhi::BufferDesc& actual = buffer->getDesc();
    CHECK_EQ(actual.byteSize, 1024ull);
    CHECK_EQ(actual.debugName, std::string("Test buffer"));
    CHECK(actual.isVertexBuffer);
    CHECK(actual.isIndexBuffer);
    CHECK(actual.canHaveRawViews);
    CHECK(buffer->getGpuVirtualAddress() != 0);
    CHECK(buffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource).pointer != nullptr);
}

NVRHI_TEST_CASE("constant buffers align their storage to 256 bytes")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BufferHandle buffer = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(100)
        .setIsConstantBuffer(true));
    REQUIRE(buffer != nullptr);
    CHECK_EQ(buffer->getDesc().byteSize, 256ull);

    nvrhi::BufferHandle exact = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(512)
        .setIsConstantBuffer(true));
    REQUIRE(exact != nullptr);
    CHECK_EQ(exact->getDesc().byteSize, 512ull);
}

NVRHI_TEST_CASE("volatile buffers have no backing memory")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BufferDesc desc = nvrhi::utils::CreateVolatileConstantBufferDesc(256,
        "Volatile constants", 3);
    nvrhi::BufferHandle buffer = testDevice->createBuffer(desc);
    REQUIRE(buffer != nullptr);
    CHECK(buffer->getDesc().isVolatile);
    CHECK_EQ(buffer->getDesc().maxVersions, 3u);

    nvrhi::MemoryRequirements requirements = { 123, 456 };
    CHECK(!buffer->queryMemoryRequirements(requirements));
    CHECK_EQ(requirements.size, 123ull);
    CHECK_EQ(requirements.alignment, 456ull);
}

NVRHI_TEST_CASE("buffer memory requirements are reported consistently")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BufferHandle buffer = testDevice->createBuffer(
        nvrhi::BufferDesc().setByteSize(4096));
    REQUIRE(buffer != nullptr);

    const nvrhi::MemoryRequirements legacy =
        testDevice->getBufferMemoryRequirements(buffer);
    CHECK(legacy.size >= 4096);
    CHECK(legacy.alignment > 0);

    nvrhi::MemoryRequirements queried = {};
    CHECK(buffer->queryMemoryRequirements(queried));
    CHECK_EQ(queried.size, legacy.size);
    CHECK_EQ(queried.alignment, legacy.alignment);

    nvrhi::TextureHandle texture = testDevice->createTexture(
        makeColorTextureDesc(64, 64));
    REQUIRE(texture != nullptr);

    const nvrhi::MemoryRequirements textureRequirements =
        testDevice->getTextureMemoryRequirements(texture);
    CHECK(textureRequirements.size >= 64ull * 64ull * 4ull);
    CHECK(textureRequirements.alignment > 0);
}

NVRHI_TEST_CASE("virtual buffers bind to heap memory once")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BufferHandle buffer = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(65536)
        .setIsVirtual(true));
    REQUIRE(buffer != nullptr);
    CHECK(buffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource).pointer == nullptr);

    nvrhi::MemoryRequirements requirements = {};
    CHECK(buffer->queryMemoryRequirements(requirements));
    REQUIRE(requirements.size > 0);
    REQUIRE(requirements.alignment > 0);

    nvrhi::HeapHandle heap = testDevice->createHeap(nvrhi::HeapDesc()
        .setCapacity(nvrhi::align(requirements.size, 65536ull))
        .setType(nvrhi::HeapType::DeviceLocal)
        .setDebugName("Virtual buffer heap"));
    REQUIRE(heap != nullptr);
    CHECK(heap->getDesc().type == nvrhi::HeapType::DeviceLocal);

    nvrhi::MemoryRequirements heapRequirements = { 77, 88 };
    CHECK(!heap->queryMemoryRequirements(heapRequirements));
    CHECK_EQ(heapRequirements.size, 77ull);
    CHECK_EQ(heapRequirements.alignment, 88ull);

    CHECK(testDevice->bindBufferMemory(buffer, heap, 0));
    CHECK(buffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource).pointer != nullptr);
    CHECK(!testDevice->bindBufferMemory(buffer, heap, 0));

    nvrhi::BufferHandle committed = testDevice->createBuffer(
        nvrhi::BufferDesc().setByteSize(1024));
    REQUIRE(committed != nullptr);
    CHECK(!testDevice->bindBufferMemory(committed, heap, 0));
}

NVRHI_TEST_CASE("virtual textures bind to heap memory once")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc desc = makeColorTextureDesc(64, 64);
    desc.setIsVirtual(true);

    nvrhi::TextureHandle texture = testDevice->createTexture(desc);
    REQUIRE(texture != nullptr);

    nvrhi::MemoryRequirements requirements = {};
    requirements = testDevice->getTextureMemoryRequirements(texture);
    REQUIRE(requirements.size > 0);
    REQUIRE(requirements.alignment > 0);

    nvrhi::HeapHandle heap = testDevice->createHeap(nvrhi::HeapDesc()
        .setCapacity(nvrhi::align(requirements.size, 65536ull))
        .setType(nvrhi::HeapType::DeviceLocal));
    REQUIRE(heap != nullptr);

    CHECK(testDevice->bindTextureMemory(texture, heap, 0));
    CHECK(!testDevice->bindTextureMemory(texture, heap, 0));

    nvrhi::TextureHandle committed = testDevice->createTexture(makeColorTextureDesc(64, 64));
    REQUIRE(committed != nullptr);
    CHECK(!testDevice->bindTextureMemory(committed, heap, 0));
}

NVRHI_TEST_CASE("textures expose views for their declared usage")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc desc = makeColorTextureDesc(64, 32);
    desc.setDebugName("Color target");

    nvrhi::TextureHandle texture = testDevice->createTexture(desc);
    REQUIRE(texture != nullptr);

    const nvrhi::TextureDesc& actual = texture->getDesc();
    CHECK_EQ(actual.width, 64u);
    CHECK_EQ(actual.height, 32u);
    CHECK(actual.isRenderTarget);
    CHECK(texture->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource).pointer != nullptr);

    const nvrhi::Object renderTargetView =
        texture->getNativeView(nvrhi::ObjectTypes::D3D12_RenderTargetViewDescriptor);
    CHECK(renderTargetView.integer != 0);

    const nvrhi::Object shaderResourceView =
        texture->getNativeView(nvrhi::ObjectTypes::D3D12_ShaderResourceViewGpuDescriptor);
    CHECK(shaderResourceView.integer != 0);

    nvrhi::TextureHandle arrayTexture = testDevice->createTexture(
        nvrhi::TextureDesc()
            .setWidth(16)
            .setHeight(16)
            .setMipLevels(3)
            .setArraySize(6)
            .setDimension(nvrhi::TextureDimension::Texture2DArray)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setInitialState(nvrhi::ResourceStates::Common));
    REQUIRE(arrayTexture != nullptr);
    CHECK_EQ(arrayTexture->getDesc().arraySize, 6u);
    CHECK_EQ(arrayTexture->getDesc().mipLevels, 3u);

    nvrhi::TextureHandle volumeTexture = testDevice->createTexture(
        nvrhi::TextureDesc()
            .setWidth(8)
            .setHeight(8)
            .setDepth(8)
            .setDimension(nvrhi::TextureDimension::Texture3D)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setInitialState(nvrhi::ResourceStates::Common));
    REQUIRE(volumeTexture != nullptr);
    CHECK_EQ(volumeTexture->getDesc().depth, 8u);

    nvrhi::TextureHandle uavTexture = testDevice->createTexture(
        nvrhi::TextureDesc()
            .setWidth(8)
            .setHeight(8)
            .setFormat(nvrhi::Format::R32_UINT)
            .setIsUAV(true)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess));
    REQUIRE(uavTexture != nullptr);
    CHECK(uavTexture->getDesc().isUAV);
    CHECK(uavTexture->getNativeView(
        nvrhi::ObjectTypes::D3D12_UnorderedAccessViewGpuDescriptor).integer != 0);
}

NVRHI_TEST_CASE("depth textures expose a depth stencil view")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureHandle depth = testDevice->createTexture(makeDepthTextureDesc(64, 64));
    REQUIRE(depth != nullptr);
    CHECK(depth->getDesc().format == nvrhi::Format::D32);

    const nvrhi::Object depthView =
        depth->getNativeView(nvrhi::ObjectTypes::D3D12_DepthStencilViewDescriptor);
    CHECK(depthView.integer != 0);

    const nvrhi::Object readOnlyView = depth->getNativeView(
        nvrhi::ObjectTypes::D3D12_DepthStencilViewDescriptor,
        nvrhi::Format::UNKNOWN, nvrhi::AllSubresources,
        nvrhi::TextureDimension::Unknown, true);
    CHECK(readOnlyView.integer != 0);

    nvrhi::TextureHandle multisampled = testDevice->createTexture(
        nvrhi::TextureDesc()
            .setWidth(16)
            .setHeight(16)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setDimension(nvrhi::TextureDimension::Texture2DMS)
            .setSampleCount(4)
            .setIsRenderTarget(true)
            .setInitialState(nvrhi::ResourceStates::RenderTarget));
    REQUIRE(multisampled != nullptr);
    CHECK_EQ(multisampled->getDesc().sampleCount, 4u);
}

NVRHI_TEST_CASE("staging textures map a resolved slice")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc desc = makeColorTextureDesc(8, 4);
    nvrhi::StagingTextureHandle staging =
        testDevice->createStagingTexture(desc, nvrhi::CpuAccessMode::Read);
    REQUIRE(staging != nullptr);
    CHECK_EQ(staging->getDesc().width, 8u);
    CHECK(staging->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource).pointer != nullptr);

    const nvrhi::TextureSlice slice = nvrhi::TextureSlice().resolve(staging->getDesc());
    size_t rowPitch = 0;
    void* mapped = testDevice->mapStagingTexture(staging, slice,
        nvrhi::CpuAccessMode::Read, &rowPitch);
    REQUIRE(mapped != nullptr);
    CHECK(rowPitch >= 8u * 4u);
    testDevice->unmapStagingTexture(staging);
}

NVRHI_TEST_CASE("tiled textures report their tile layout")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureDesc tiledDesc;
    tiledDesc.setWidth(256)
        .setHeight(256)
        .setFormat(nvrhi::Format::RGBA8_UNORM)
        .setKeepInitialState(false)
        .setInitialState(nvrhi::ResourceStates::Common);
    tiledDesc.isTiled = true;

    nvrhi::TextureHandle texture = testDevice->createTexture(tiledDesc);
    REQUIRE(texture != nullptr);

    uint32_t numTiles = 0;
    nvrhi::PackedMipDesc packedMip = {};
    nvrhi::TileShape tileShape = {};
    uint32_t numTilings = 16;
    nvrhi::SubresourceTiling tilings[16] = {};

    testDevice->getTextureTiling(texture, &numTiles, &packedMip, &tileShape,
        &numTilings, tilings);

    CHECK(numTiles > 0);
    CHECK(tileShape.widthInTexels > 0);
    CHECK(numTilings >= 1);
    CHECK_EQ(packedMip.numStandardMips + packedMip.numPackedMips, 1u);
}

NVRHI_TEST_CASE("samplers round-trip their description")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::SamplerDesc desc;
    desc.setAllFilters(true)
        .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
        .setBorderColor(nvrhi::Color(0.f, 1.f, 0.f, 1.f))
        .setMaxAnisotropy(8.f)
        .setMipBias(-0.5f)
        .setReductionType(nvrhi::SamplerReductionType::Comparison);

    nvrhi::SamplerHandle sampler = testDevice->createSampler(desc);
    REQUIRE(sampler != nullptr);

    const nvrhi::SamplerDesc& actual = sampler->getDesc();
    CHECK(actual.minFilter && actual.magFilter && actual.mipFilter);
    CHECK(actual.addressU == nvrhi::SamplerAddressMode::Wrap);
    CHECK(actual.addressV == nvrhi::SamplerAddressMode::Wrap);
    CHECK(actual.addressW == nvrhi::SamplerAddressMode::Wrap);
    CHECK(actual.borderColor == nvrhi::Color(0.f, 1.f, 0.f, 1.f));
    CHECK_EQ(actual.maxAnisotropy, 8.f);
    CHECK_EQ(actual.mipBias, -0.5f);
    CHECK(actual.reductionType == nvrhi::SamplerReductionType::Comparison);
    CHECK(sampler->getNativeObject(nvrhi::ObjectTypes::SharedHandle).integer == 0);
}

NVRHI_TEST_CASE("input layouts round-trip their attributes")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::VertexAttributeDesc attributes[2];
    attributes[0].setName("POSITION")
        .setFormat(nvrhi::Format::RGB32_FLOAT)
        .setBufferIndex(0)
        .setOffset(0)
        .setElementStride(32);
    attributes[1].setName("TEXCOORD")
        .setFormat(nvrhi::Format::RG32_FLOAT)
        .setBufferIndex(0)
        .setOffset(12)
        .setElementStride(32)
        .setIsInstanced(false);

    nvrhi::InputLayoutHandle layout =
        testDevice->createInputLayout(attributes, 2, nullptr);
    REQUIRE(layout != nullptr);
    CHECK_EQ(layout->getNumAttributes(), 2u);

    const nvrhi::VertexAttributeDesc* position = layout->getAttributeDesc(0);
    REQUIRE(position != nullptr);
    CHECK_EQ(position->name, std::string("POSITION"));
    CHECK(position->format == nvrhi::Format::RGB32_FLOAT);
    CHECK_EQ(position->elementStride, 32u);

    const nvrhi::VertexAttributeDesc* texcoord = layout->getAttributeDesc(1);
    REQUIRE(texcoord != nullptr);
    CHECK_EQ(texcoord->name, std::string("TEXCOORD"));
    CHECK_EQ(texcoord->offset, 12u);
    CHECK(layout->getAttributeDesc(2) == nullptr);
}

NVRHI_TEST_CASE("binding layouts describe their items")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BindingLayoutDesc desc;
    desc.setVisibility(nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel)
        .setRegisterSpace(2)
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
        .addItem(nvrhi::BindingLayoutItem::Sampler(1))
        .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(2));

    nvrhi::BindingLayoutHandle layout = testDevice->createBindingLayout(desc);
    REQUIRE(layout != nullptr);

    const nvrhi::BindingLayoutDesc* actual = layout->getDesc();
    REQUIRE(actual != nullptr);
    CHECK(actual->visibility == (nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel));
    CHECK_EQ(actual->registerSpace, 2u);
    CHECK_EQ(actual->bindings.size(), size_t(3));
    CHECK(actual->bindings[0].type == nvrhi::ResourceType::Texture_SRV);
    CHECK(actual->bindings[1].type == nvrhi::ResourceType::Sampler);
    CHECK(actual->bindings[2].type == nvrhi::ResourceType::ConstantBuffer);
    CHECK(layout->getBindlessDesc() == nullptr);
}

NVRHI_TEST_CASE("bindless layouts describe their capacity")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BindlessLayoutDesc desc;
    desc.setVisibility(nvrhi::ShaderType::Pixel)
        .setFirstSlot(0)
        .setMaxCapacity(64)
        .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_SRV(0))
        .setLayoutType(nvrhi::BindlessLayoutDesc::LayoutType::Immutable);

    nvrhi::BindingLayoutHandle layout = testDevice->createBindlessLayout(desc);
    REQUIRE(layout != nullptr);

    const nvrhi::BindlessLayoutDesc* actual = layout->getBindlessDesc();
    REQUIRE(actual != nullptr);
    CHECK_EQ(actual->maxCapacity, 64u);
    CHECK_EQ(actual->registerSpaces.size(), size_t(1));
    CHECK(actual->registerSpaces[0].type == nvrhi::ResourceType::Texture_SRV);
    CHECK(layout->getDesc() == nullptr);
}

NVRHI_TEST_CASE("binding sets reference their layout")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureHandle texture = testDevice->createTexture(
        makeColorTextureDesc(16, 16));
    REQUIRE(texture != nullptr);

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.setVisibility(nvrhi::ShaderType::Pixel)
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0));

    nvrhi::BindingLayoutHandle layout = testDevice->createBindingLayout(layoutDesc);
    REQUIRE(layout != nullptr);

    nvrhi::BindingSetDesc setDesc;
    setDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(0, texture));

    nvrhi::BindingSetHandle bindingSet = testDevice->createBindingSet(setDesc, layout);
    REQUIRE(bindingSet != nullptr);
    CHECK(bindingSet->getLayout() == layout.Get());

    const nvrhi::BindingSetDesc* actual = bindingSet->getDesc();
    REQUIRE(actual != nullptr);
    CHECK_EQ(actual->bindings.size(), size_t(1));
    CHECK(actual->bindings[0].resourceHandle == texture.Get());
}

NVRHI_TEST_CASE("descriptor tables grow and accept writes")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TextureHandle texture = testDevice->createTexture(
        makeColorTextureDesc(16, 16));
    REQUIRE(texture != nullptr);

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.setVisibility(nvrhi::ShaderType::Pixel)
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0));
    nvrhi::BindingLayoutHandle layout = testDevice->createBindingLayout(layoutDesc);
    REQUIRE(layout != nullptr);

    nvrhi::DescriptorTableHandle table = testDevice->createDescriptorTable(layout);
    REQUIRE(table != nullptr);
    CHECK_EQ(table->getCapacity(), 0u);
    CHECK(table->getLayout() == layout.Get());

    testDevice->resizeDescriptorTable(table, 16);
    CHECK_EQ(table->getCapacity(), 16u);

    CHECK(testDevice->writeDescriptorTable(table,
        nvrhi::BindingSetItem::Texture_SRV(0, texture)));

    const uint32_t firstDescriptor = table->getFirstDescriptorIndexInHeap();
    testDevice->resizeDescriptorTable(table, 8);
    CHECK_EQ(table->getCapacity(), 8u);
    CHECK_EQ(table->getFirstDescriptorIndexInHeap(), firstDescriptor);

    testDevice->resizeDescriptorTable(table, 24, false);
    CHECK_EQ(table->getCapacity(), 24u);
    CHECK(table->getFirstDescriptorIndexInHeap() < 65536u);
}

NVRHI_TEST_CASE("command lists open close and execute")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::CommandListParameters parameters;
    parameters.setQueueType(nvrhi::CommandQueue::Graphics)
        .setUploadChunkSize(4096)
        .setScratchChunkSize(4096);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList(parameters);
    REQUIRE(commandList != nullptr);
    CHECK(commandList->getDevice() == testDevice);
    CHECK(commandList->getDesc().queueType == nvrhi::CommandQueue::Graphics);
    CHECK_EQ(commandList->getDesc().uploadChunkSize, size_t(4096));

    commandList->open();
    commandList->clearState();
    commandList->beginMarker("Test range");
    commandList->endMarker();
    commandList->insertGlobalSyncBarrier();
    commandList->setEnableAutomaticBarriers(false);
    commandList->setEnableAutomaticBarriers(true);
    commandList->close();

    testDevice->executeCommandList(commandList);
    CHECK(testDevice->waitForIdle());
}

NVRHI_TEST_CASE("command lists execute on compute and copy queues")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::CommandListHandle compute = testDevice->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Compute));
    nvrhi::CommandListHandle copy = testDevice->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Copy));
    REQUIRE(compute != nullptr);
    REQUIRE(copy != nullptr);

    compute->open();
    compute->close();
    const uint64_t computeInstance =
        testDevice->executeCommandList(compute, nvrhi::CommandQueue::Compute);

    copy->open();
    copy->close();
    testDevice->executeCommandList(copy, nvrhi::CommandQueue::Copy);

    testDevice->queueWaitForCommandList(nvrhi::CommandQueue::Graphics,
        nvrhi::CommandQueue::Compute, computeInstance);

    nvrhi::CommandListHandle graphics = testDevice->createCommandList();
    REQUIRE(graphics != nullptr);
    graphics->open();
    graphics->close();
    testDevice->executeCommandList(graphics);
    CHECK(testDevice->waitForIdle());
}

NVRHI_TEST_CASE("command lists track resource states")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BufferHandle buffer = testDevice->createBuffer(nvrhi::BufferDesc()
        .setByteSize(1024)
        .setInitialState(nvrhi::ResourceStates::Common));
    REQUIRE(buffer != nullptr);

    nvrhi::TextureHandle texture = testDevice->createTexture(makeColorTextureDesc(16, 16));
    REQUIRE(texture != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();

    CHECK(commandList->getBufferState(buffer) == nvrhi::ResourceStates::Unknown);
    commandList->beginTrackingBufferState(buffer, nvrhi::ResourceStates::Common);
    CHECK(commandList->getBufferState(buffer) == nvrhi::ResourceStates::Common);

    commandList->setBufferState(buffer, nvrhi::ResourceStates::CopyDest);
    CHECK(commandList->getBufferState(buffer) == nvrhi::ResourceStates::CopyDest);
    commandList->commitBarriers();
    commandList->setBufferState(buffer, nvrhi::ResourceStates::Common);
    CHECK(commandList->getBufferState(buffer) == nvrhi::ResourceStates::Common);

    commandList->beginTrackingTextureState(texture, nvrhi::AllSubresources,
        nvrhi::ResourceStates::RenderTarget);
    CHECK(commandList->getTextureSubresourceState(texture, 0, 0) ==
        nvrhi::ResourceStates::RenderTarget);
    commandList->setTextureState(texture, nvrhi::AllSubresources,
        nvrhi::ResourceStates::CopySource);
    CHECK(commandList->getTextureSubresourceState(texture, 0, 0) ==
        nvrhi::ResourceStates::CopySource);

    commandList->commitBarriers();
    commandList->setPermanentTextureState(texture, nvrhi::ResourceStates::RenderTarget);
    commandList->setPermanentBufferState(buffer, nvrhi::ResourceStates::Common);
    commandList->close();

    testDevice->executeCommandList(commandList);
    CHECK(testDevice->waitForIdle());
    CHECK(commandList->getTextureSubresourceState(texture, 0, 0) ==
        nvrhi::ResourceStates::RenderTarget);
}

NVRHI_TEST_CASE("command lists report internal memory usage")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BufferHandle buffer = testDevice->createBuffer(
        nvrhi::BufferDesc().setByteSize(256));
    REQUIRE(buffer != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();

    uint8_t data[256] = {};
    commandList->writeBuffer(buffer, data, sizeof(data));

    const nvrhi::CommandListMemoryUsage usage = commandList->getInternalMemoryUsage();
    CHECK(usage.uploadBytes > 0);

    commandList->close();
    testDevice->executeCommandList(commandList);
    CHECK(testDevice->waitForIdle());
}

NVRHI_TEST_CASE("event queries complete after execution")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::EventQueryHandle query = testDevice->createEventQuery();
    REQUIRE(query != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->close();
    testDevice->executeCommandList(commandList);

    testDevice->setEventQuery(query, nvrhi::CommandQueue::Graphics);
    testDevice->waitEventQuery(query);
    CHECK(testDevice->pollEventQuery(query));

    testDevice->resetEventQuery(query);
    CHECK(!testDevice->pollEventQuery(query));
}

NVRHI_TEST_CASE("timer queries return elapsed time")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::TimerQueryHandle query = testDevice->createTimerQuery();
    REQUIRE(query != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->beginTimerQuery(query);
    commandList->endTimerQuery(query);
    commandList->close();
    testDevice->executeCommandList(commandList);

    CHECK(testDevice->waitForIdle());
    CHECK(testDevice->pollTimerQuery(query));
    CHECK(testDevice->getTimerQueryTime(query) >= 0.f);

    testDevice->resetTimerQuery(query);
    CHECK_EQ(testDevice->getTimerQueryTime(query), 0.f);
}

NVRHI_TEST_CASE("pipeline statistics queries run")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::PipelineStatisticsQueryHandle query =
        testDevice->createPipelineStatisticsQuery();
    REQUIRE(query != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->beginPipelineStatisticsQuery(query);
    commandList->endPipelineStatisticsQuery(query);
    commandList->close();
    testDevice->executeCommandList(commandList);
    CHECK(testDevice->waitForIdle());

    CHECK(testDevice->pollPipelineStatisticsQuery(query));
    const nvrhi::PipelineStatistics statistics =
        testDevice->getPipelineStatistics(query);
    CHECK_EQ(statistics.VSInvocations, 0ull);
    CHECK_EQ(statistics.CSInvocations, 0ull);

    testDevice->resetPipelineStatisticsQuery(query);
    CHECK(!testDevice->pollPipelineStatisticsQuery(query));
}

NVRHI_TEST_CASE("native buffers can be imported")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();
    ID3D12Device* native = nvrhi_test::nativeDevice();
    REQUIRE(native != nullptr);

    D3D12_HEAP_PROPERTIES heapProperties = {};
    heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Width = 256;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    REQUIRE(SUCCEEDED(native->CreateCommittedResource(&heapProperties,
        D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&resource))));

    nvrhi::BufferHandle buffer = testDevice->createHandleForNativeBuffer(
        nvrhi::ObjectTypes::D3D12_Resource, resource.Get(),
        nvrhi::BufferDesc().setByteSize(256));
    REQUIRE(buffer != nullptr);
    CHECK_EQ(buffer->getDesc().byteSize, 256ull);
    CHECK(buffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource).pointer ==
        resource.Get());
}

NVRHI_TEST_CASE("native textures can be imported")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();
    ID3D12Device* native = nvrhi_test::nativeDevice();
    REQUIRE(native != nullptr);

    D3D12_HEAP_PROPERTIES heapProperties = {};
    heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Width = 16;
    resourceDesc.Height = 16;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    REQUIRE(SUCCEEDED(native->CreateCommittedResource(&heapProperties,
        D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&resource))));

    nvrhi::TextureHandle texture = testDevice->createHandleForNativeTexture(
        nvrhi::ObjectTypes::D3D12_Resource, resource.Get(),
        nvrhi::TextureDesc()
            .setWidth(16)
            .setHeight(16)
            .setFormat(nvrhi::Format::RGBA8_UNORM)
            .setInitialState(nvrhi::ResourceStates::Common));
    REQUIRE(texture != nullptr);
    CHECK_EQ(texture->getDesc().width, 16u);
    CHECK(texture->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource).pointer ==
        resource.Get());
}

NVRHI_TEST_CASE("resources are reference counted through handles")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BufferHandle buffer = testDevice->createBuffer(
        nvrhi::BufferDesc().setByteSize(256));
    REQUIRE(buffer != nullptr);

    const unsigned long initial = buffer->GetRefCount();
    REQUIRE(initial >= 1);

    {
        nvrhi::BufferHandle copy = buffer;
        CHECK_EQ(buffer->GetRefCount(), initial + 1);

        nvrhi::BufferHandle moved = std::move(copy);
        CHECK(copy == nullptr);
        CHECK_EQ(buffer->GetRefCount(), initial + 1);
        CHECK(moved.Get() == buffer.Get());
    }

    CHECK_EQ(buffer->GetRefCount(), initial);
}

NVRHI_TEST_CASE("command list lifetime trackers run garbage collection")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::CommandListLifetimeTrackerHandle tracker =
        testDevice->createCommandListLifetimeTracker(nvrhi::CommandQueue::Graphics);
    REQUIRE(tracker != nullptr);
    tracker->runGarbageCollection();

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    commandList->open();
    commandList->close();
    testDevice->executeCommandList(commandList);
    CHECK(testDevice->waitForIdle());

    tracker->runGarbageCollection();
    testDevice->runGarbageCollection();

    nvrhi::CommandListLifetimeTrackerHandle copyTracker =
        testDevice->createCommandListLifetimeTracker(nvrhi::CommandQueue::Copy);
    REQUIRE(copyTracker != nullptr);
    copyTracker->runGarbageCollection();
}

NVRHI_TEST_CASE("root signatures can be built directly")
{
    REQUIRE_DEVICE();
    nvrhi::d3d12::IDevice* testDevice = nvrhi_test::d3d12Device();
    REQUIRE(testDevice != nullptr);

    nvrhi::BindingLayoutVector emptyLayouts;
    nvrhi::d3d12::RootSignatureHandle empty =
        testDevice->buildRootSignature(emptyLayouts, false, false);
    REQUIRE(empty != nullptr);
    CHECK(empty->getNativeObject(
        nvrhi::ObjectTypes::D3D12_RootSignature).pointer != nullptr);

    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.setVisibility(nvrhi::ShaderType::Pixel)
        .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0))
        .addItem(nvrhi::BindingLayoutItem::Sampler(0));
    nvrhi::BindingLayoutHandle layout = testDevice->createBindingLayout(layoutDesc);
    REQUIRE(layout != nullptr);

    nvrhi::BindingLayoutVector layouts;
    layouts.push_back(layout);
    nvrhi::d3d12::RootSignatureHandle bound =
        testDevice->buildRootSignature(layouts, false, false);
    REQUIRE(bound != nullptr);
    CHECK(bound->getNativeObject(
        nvrhi::ObjectTypes::D3D12_RootSignature).pointer != nullptr);

    nvrhi::BindingLayoutDesc localDesc;
    localDesc.setVisibility(nvrhi::ShaderType::All)
        .addItem(nvrhi::BindingLayoutItem::PushConstants(0, 16));
    nvrhi::BindingLayoutHandle localLayout = testDevice->createBindingLayout(localDesc);
    REQUIRE(localLayout != nullptr);

    nvrhi::BindingLayoutVector localLayouts;
    localLayouts.push_back(localLayout);
    nvrhi::d3d12::RootSignatureHandle local =
        testDevice->buildRootSignature(localLayouts, false, true);
    REQUIRE(local != nullptr);
    CHECK(local->getNativeObject(
        nvrhi::ObjectTypes::D3D12_RootSignature).pointer != nullptr);
}

NVRHI_TEST_CASE("D3D12 command list extensions allocate upload memory")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::BufferHandle buffer = testDevice->createBuffer(
        nvrhi::BufferDesc().setByteSize(256));
    REQUIRE(buffer != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);
    nvrhi::d3d12::ICommandList* d3d12List =
        static_cast<nvrhi::d3d12::ICommandList*>(commandList.Get());
    REQUIRE(d3d12List != nullptr);

    commandList->open();
    CHECK(d3d12List->commitDescriptorHeaps());
    CHECK(!d3d12List->commitDescriptorHeaps());
    CHECK_EQ(d3d12List->getBufferGpuVA(buffer), buffer->getGpuVirtualAddress());

    void* cpuAddress = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS gpuAddress = 0;
    REQUIRE(d3d12List->allocateUploadBuffer(64, &cpuAddress, &gpuAddress));
    REQUIRE(cpuAddress != nullptr);
    CHECK(gpuAddress != 0);
    std::memset(cpuAddress, 0x5A, 64);

    d3d12List->updateGraphicsVolatileBuffers();
    d3d12List->updateComputeVolatileBuffers();

    commandList->close();
    testDevice->executeCommandList(commandList);
    CHECK(testDevice->waitForIdle());
}

NVRHI_TEST_CASE("executeCommandLists submits several lists at once")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    nvrhi::CommandListHandle first = testDevice->createCommandList();
    nvrhi::CommandListHandle second = testDevice->createCommandList();
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);

    first->open();
    first->close();
    second->open();
    second->close();

    nvrhi::ICommandList* lists[2] = { first.Get(), second.Get() };
    const uint64_t instance = testDevice->executeCommandLists(lists, 2);
    CHECK(instance > 0);
    CHECK(testDevice->waitForIdle());
}

NVRHI_TEST_CASE("cooperative vector queries are consistent")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();

    const nvrhi::coopvec::DeviceFeatures features = testDevice->queryCoopVecFeatures();

    nvrhi::coopvec::MatMulFormatCombo combination = {};
    combination.inputType = nvrhi::coopvec::DataType::Float16;
    combination.inputInterpretation = nvrhi::coopvec::DataType::Float16;
    combination.matrixInterpretation = nvrhi::coopvec::DataType::Float16;
    combination.biasInterpretation = nvrhi::coopvec::DataType::Float16;
    combination.outputType = nvrhi::coopvec::DataType::Float16;
    combination.transposeSupported = false;

    const nvrhi::coopvec::MatMulFormatSupport matMul =
        testDevice->queryCoopVecMatMulFormatSupport(combination);
    if (matMul.supported)
        CHECK(!features.matMulFormats.empty());

    const nvrhi::coopvec::TrainingFormatSupport training =
        testDevice->queryCoopVecTrainingFormatSupport(nvrhi::coopvec::DataType::Float16);
    CHECK_EQ(training.bufferTrainingSupported,
        training.threadOuterProductSupported && training.bufferAccumulateStoreSupported);

    const size_t matrixSize = testDevice->getCoopVecMatrixSize(
        nvrhi::coopvec::DataType::Float32, nvrhi::coopvec::MatrixLayout::RowMajor, 8, 16);
    if (testDevice->queryFeatureSupport(nvrhi::Feature::CooperativeVectorInferencing))
        CHECK(matrixSize > 0);
}

NVRHI_TEST_CASE("sampler feedback textures pair with their source")
{
    REQUIRE_DEVICE();
    nvrhi::IDevice* testDevice = nvrhi_test::device();
    SKIP_IF(!testDevice->queryFeatureSupport(nvrhi::Feature::SamplerFeedback),
        "sampler feedback is not supported");

    nvrhi::TextureDesc pairedDesc;
    pairedDesc.setWidth(64)
        .setHeight(64)
        .setFormat(nvrhi::Format::RGBA8_UNORM)
        .setIsUAV(true)
        .setInitialState(nvrhi::ResourceStates::UnorderedAccess);

    nvrhi::TextureHandle paired = testDevice->createTexture(pairedDesc);
    REQUIRE(paired != nullptr);

    nvrhi::SamplerFeedbackTextureDesc desc;
    desc.samplerFeedbackFormat = nvrhi::SamplerFeedbackFormat::MipRegionUsedOpaque;
    desc.samplerFeedbackMipRegionX = 4;
    desc.samplerFeedbackMipRegionY = 4;
    desc.samplerFeedbackMipRegionZ = 1;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.keepInitialState = true;

    nvrhi::SamplerFeedbackTextureHandle feedback =
        testDevice->createSamplerFeedbackTexture(paired, desc);
    REQUIRE(feedback != nullptr);

    CHECK(feedback->getDesc().samplerFeedbackFormat ==
        nvrhi::SamplerFeedbackFormat::MipRegionUsedOpaque);
    CHECK_EQ(feedback->getDesc().samplerFeedbackMipRegionX, 4u);
    CHECK_EQ(feedback->getDesc().samplerFeedbackMipRegionY, 4u);
    CHECK(feedback->getPairedTexture().Get() == paired.Get());
    CHECK(feedback->getNativeObject(
        nvrhi::ObjectTypes::D3D12_Resource).pointer != nullptr);

    nvrhi::CommandListHandle commandList = testDevice->createCommandList();
    REQUIRE(commandList != nullptr);

    nvrhi::BufferDesc decodeDesc;
    decodeDesc.byteSize = 64 * 64;
    decodeDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
    decodeDesc.initialState = nvrhi::ResourceStates::ResolveDest;
    decodeDesc.debugName = "Sampler feedback decode buffer";
    nvrhi::BufferHandle decoded = testDevice->createBuffer(decodeDesc);
    REQUIRE(decoded != nullptr);

    commandList->open();
    commandList->clearSamplerFeedbackTexture(feedback);
    commandList->setSamplerFeedbackTextureState(feedback,
        nvrhi::ResourceStates::UnorderedAccess);
    commandList->decodeSamplerFeedbackTexture(decoded, feedback, nvrhi::Format::R8_UINT);
    commandList->close();
    testDevice->executeCommandList(commandList);
    CHECK(testDevice->waitForIdle());

    void* decodedData = testDevice->mapBuffer(decoded, nvrhi::CpuAccessMode::Read);
    REQUIRE(decodedData != nullptr);
    testDevice->unmapBuffer(decoded);

    auto* nativeResource = static_cast<ID3D12Resource*>(
        feedback->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource).pointer);
    REQUIRE(nativeResource != nullptr);

    Microsoft::WRL::ComPtr<ID3D12Resource2> native2;
    REQUIRE(SUCCEEDED(nativeResource->QueryInterface(IID_PPV_ARGS(&native2))));

    nvrhi::SamplerFeedbackTextureHandle imported =
        testDevice->createSamplerFeedbackForNativeTexture(
            nvrhi::ObjectTypes::D3D12_Resource, native2.Get(), paired);
    REQUIRE(imported != nullptr);
    CHECK(imported->getPairedTexture().Get() == paired.Get());
}
