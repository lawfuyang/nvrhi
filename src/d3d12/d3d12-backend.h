/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <nvrhi/d3d12.h>

#ifndef NVRHI_D3D12_WITH_NVAPI
#define NVRHI_D3D12_WITH_NVAPI 0
#endif

#if NVRHI_D3D12_WITH_NVAPI
#include <dxgi.h>
#include <nvapi.h>
#endif

#include <nvrhi/common/aftermath.h>
#if NVRHI_WITH_AFTERMATH
#include <GFSDK_Aftermath.h>
#endif

// There's no version check available in the nvapi header,
// instead to check if the NvAPI linked is OMM compatible version (>520) we look for one of the defines it adds...
#if NVRHI_D3D12_WITH_NVAPI && defined(NVAPI_GET_RAYTRACING_OPACITY_MICROMAP_ARRAY_PREBUILD_INFO_PARAMS_VER)
#define NVRHI_WITH_NVAPI_OPACITY_MICROMAP (1)
#else
#define NVRHI_WITH_NVAPI_OPACITY_MICROMAP (0)
#endif

// ... same for DMM compatible versions (>=535) we look for one of the defines it adds
#if NVRHI_D3D12_WITH_NVAPI && defined(NVAPI_GET_RAYTRACING_DISPLACEMENT_MICROMAP_ARRAY_PREBUILD_INFO_PARAMS_VER)
#define NVRHI_WITH_NVAPI_DISPLACEMENT_MICROMAP (1)
#else
#define NVRHI_WITH_NVAPI_DISPLACEMENT_MICROMAP (0)
#endif

#if NVRHI_D3D12_WITH_NVAPI && defined(NVAPI_GET_RAYTRACING_MULTI_INDIRECT_CLUSTER_OPERATION_REQUIREMENTS_INFO_PARAMS_VER)
#define NVRHI_WITH_NVAPI_CLUSTERS (1)
#else
#define NVRHI_WITH_NVAPI_CLUSTERS (0)
#endif

// Line-Swept Spheres were added in NVAPI SDK 572.18
#if NVRHI_D3D12_WITH_NVAPI && (NVAPI_SDK_VERSION >= 57218)
#define NVRHI_WITH_NVAPI_LSS (1)
#else
#define NVRHI_WITH_NVAPI_LSS (0)
#endif

#if D3D12_PREVIEW_SDK_VERSION >= 717
#define NVRHI_D3D12_WITH_COOPVEC (1)
#else
#define NVRHI_D3D12_WITH_COOPVEC (0)
#endif

#include <bitset>
#include <memory>
#include <queue>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <nvrhi/common/resourcebindingmap.h>
#include <nvrhi/utils.h>
#include "../common/state-tracking.h"
#include "../common/dxgi-format.h"
#include "../common/versioning.h"

#ifdef NVRHI_WITH_RTXMU
#include <rtxmu/D3D12AccelStructManager.h>
#endif

// [rlaw] BEGIN
#ifdef NVRHI_D3D12_WITH_D3D12MA

#include "../thirdparty/D3D12MA/include/D3D12MemAlloc.h"
#endif
// [rlaw] END

namespace nvrhi::d3d12
{
    class RootSignature;
    class Buffer;
    class CommandList;
    struct Context;

    typedef uint32_t RootParameterIndex;
    typedef uint32_t OptionalResourceState; // D3D12_RESOURCE_STATES + unknown value

    constexpr RootParameterIndex c_InvalidRootParameterIndex = ~0u; // Used to skip mutable descriptor set
    constexpr DescriptorIndex c_InvalidDescriptorIndex = ~0u;
    constexpr OptionalResourceState c_ResourceStateUnknown = ~0u;
    
    D3D12_SHADER_VISIBILITY convertShaderStage(ShaderType s);
    D3D12_BLEND convertBlendValue(BlendFactor value);
    D3D12_BLEND_OP convertBlendOp(BlendOp value);
    D3D12_STENCIL_OP convertStencilOp(StencilOp value);
    D3D12_COMPARISON_FUNC convertComparisonFunc(ComparisonFunc value);
    D3D_PRIMITIVE_TOPOLOGY convertPrimitiveType(PrimitiveType pt, uint32_t controlPoints);
    D3D12_TEXTURE_ADDRESS_MODE convertSamplerAddressMode(SamplerAddressMode mode);
    UINT convertSamplerReductionType(SamplerReductionType reductionType);
    D3D12_SHADING_RATE convertPixelShadingRate(VariableShadingRate shadingRate);
    D3D12_SHADING_RATE_COMBINER convertShadingRateCombiner(ShadingRateCombiner combiner);
#if NVRHI_D3D12_WITH_COOPVEC
    D3D12_LINEAR_ALGEBRA_DATATYPE convertCoopVecDataType(coopvec::DataType type);
    coopvec::DataType convertCoopVecDataType(D3D12_LINEAR_ALGEBRA_DATATYPE type);
    D3D12_LINEAR_ALGEBRA_MATRIX_LAYOUT convertCoopVecMatrixLayout(coopvec::MatrixLayout layout);
#endif

    void WaitForFence(ID3D12Fence* fence, uint64_t value, HANDLE event);
    uint32_t calcSubresource(uint32_t MipSlice, uint32_t ArraySlice, uint32_t PlaneSlice, uint32_t MipLevels, uint32_t ArraySize);
    void TranslateBlendState(const BlendState& inState, D3D12_BLEND_DESC& outState);
    void TranslateDepthStencilState(const DepthStencilState& inState, D3D12_DEPTH_STENCIL_DESC& outState);
    void TranslateRasterizerState(const RasterState& inState, D3D12_RASTERIZER_DESC& outState);
    
    struct Context
    {
        RefCountPtr<ID3D12Device> device;
        RefCountPtr<ID3D12Device2> device2;
        RefCountPtr<ID3D12Device5> device5;
        RefCountPtr<ID3D12Device8> device8;
#if NVRHI_D3D12_WITH_COOPVEC
        RefCountPtr<ID3D12DevicePreview> devicePreview;
#endif
#ifdef NVRHI_WITH_RTXMU
        std::unique_ptr<rtxmu::DxAccelStructManager> rtxMemUtil;
#endif

        RefCountPtr<ID3D12CommandSignature> drawIndirectSignature;
        RefCountPtr<ID3D12CommandSignature> drawIndexedIndirectSignature;
        RefCountPtr<ID3D12CommandSignature> dispatchIndirectSignature;
        RefCountPtr<ID3D12CommandSignature> dispatchMeshIndirectSignature;  // [rlaw]: support dispatchMeshIndirect
        RefCountPtr<ID3D12QueryHeap> timerQueryHeap;
        RefCountPtr<Buffer> timerQueryResolveBuffer;

        // [rlaw] BEGIN: Pipeline Query support
        RefCountPtr<ID3D12QueryHeap> pipelineStatisticsQueryHeap;
        RefCountPtr<Buffer> pipelineStatisticsQueryResolveBuffer;
        // [rlaw] END: Pipeline Query support

        bool logBufferLifetime = false;
        IMessageCallback* messageCallback = nullptr;
        void error(const std::string& message) const;
        void info(const std::string& message) const;
    };

    class StaticDescriptorHeap : public IDescriptorHeap
    {
    private:
        const Context& m_Context;
        RefCountPtr<ID3D12DescriptorHeap> m_Heap;
        RefCountPtr<ID3D12DescriptorHeap> m_ShaderVisibleHeap;
        D3D12_DESCRIPTOR_HEAP_TYPE m_HeapType = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        D3D12_CPU_DESCRIPTOR_HANDLE m_StartCpuHandle = { 0 };
        D3D12_CPU_DESCRIPTOR_HANDLE m_StartCpuHandleShaderVisible = { 0 };
        D3D12_GPU_DESCRIPTOR_HANDLE m_StartGpuHandleShaderVisible = { 0 };
        uint32_t m_Stride = 0;
        uint32_t m_NumDescriptors = 0;
        std::vector<bool> m_AllocatedDescriptors;
        DescriptorIndex m_SearchStart = 0;
        uint32_t m_NumAllocatedDescriptors = 0;
        std::mutex m_Mutex;

        HRESULT Grow(uint32_t minRequiredSize);
    public:
        explicit StaticDescriptorHeap(const Context& context);

        HRESULT allocateResources(D3D12_DESCRIPTOR_HEAP_TYPE heapType, uint32_t numDescriptors, bool shaderVisible);
        void copyToShaderVisibleHeap(DescriptorIndex index, uint32_t count = 1);
        
        DescriptorIndex allocateDescriptors(uint32_t count) override;
        DescriptorIndex allocateDescriptor() override;
        void releaseDescriptors(DescriptorIndex baseIndex, uint32_t count) override;
        void releaseDescriptor(DescriptorIndex index) override;
        D3D12_CPU_DESCRIPTOR_HANDLE getCpuHandle(DescriptorIndex index) override;
        D3D12_CPU_DESCRIPTOR_HANDLE getCpuHandleShaderVisible(DescriptorIndex index) override;
        D3D12_GPU_DESCRIPTOR_HANDLE getGpuHandle(DescriptorIndex index) override;
        [[nodiscard]] ID3D12DescriptorHeap* getHeap() const override;
        [[nodiscard]] ID3D12DescriptorHeap* getShaderVisibleHeap() const override;

        friend class CommandList; // [rlaw]: CommandList needs to access m_Mutex
    };

    class DeviceResources
    {
    public:
        StaticDescriptorHeap renderTargetViewHeap;
        StaticDescriptorHeap depthStencilViewHeap;
        StaticDescriptorHeap shaderResourceViewHeap;
        StaticDescriptorHeap samplerHeap;
        utils::BitSetAllocator timerQueries;
        utils::BitSetAllocator pipelineStatisticsQueries; // [rlaw]: Pipeline Query support
#ifdef NVRHI_WITH_RTXMU
        std::mutex asListMutex;
        std::vector<uint64_t> asBuildsCompleted;
#endif

        // The cache does not own the RS objects, so store weak references
        std::unordered_map<size_t, RootSignature*> rootsigCache;

        explicit DeviceResources(const Context& context, const DeviceDesc& desc);

        uint8_t getFormatPlaneCount(DXGI_FORMAT format);

    private:
        const Context& m_Context;
        std::unordered_map<DXGI_FORMAT, uint8_t> m_DxgiFormatPlaneCounts;
    };


    class Shader : public RefCounter<IShader>
    {
    public:
        ShaderDesc desc;
        std::vector<char> bytecode;
    #if NVRHI_D3D12_WITH_NVAPI
        std::vector<NVAPI_D3D12_PSO_EXTENSION_DESC*> extensions;
        std::vector<NV_CUSTOM_SEMANTIC> customSemantics;
        std::vector<uint32_t> coordinateSwizzling;
    #endif
        
        const ShaderDesc& getDesc() const override { return desc; }
        void getBytecode(const void** ppBytecode, size_t* pSize) const override;
    };

    class ShaderLibrary;

    class ShaderLibraryEntry : public RefCounter<IShader>
    {
    public:
        ShaderDesc desc;
        RefCountPtr<IShaderLibrary> library;

        ShaderLibraryEntry(IShaderLibrary* pLibrary, const char* entryName, ShaderType shaderType)
            : library(pLibrary)
        {
            desc.shaderType = shaderType;
            desc.entryName = entryName;
        }

        const ShaderDesc& getDesc() const override { return desc; }
        void getBytecode(const void** ppBytecode, size_t* pSize) const override;
    };

    class ShaderLibrary : public RefCounter<IShaderLibrary>
    {
    public:
        std::vector<char> bytecode;

        void getBytecode(const void** ppBytecode, size_t* pSize) const override;
        ShaderHandle getShader(const char* entryName, ShaderType shaderType) override;
    };

    class Heap : public RefCounter<IHeap>
    {
    public:
        HeapDesc desc;
        RefCountPtr<ID3D12Heap> heap;

    // [rlaw] BEGIN
    #ifdef NVRHI_D3D12_WITH_D3D12MA
        ~Heap() override;
        D3D12MA::Allocation* m_Allocation = nullptr;
    #endif
    // [rlaw] END

        const HeapDesc& getDesc() override { return desc; }
    };

    class Texture : public RefCounter<ITexture>, public TextureStateExtension
    {
    public:
        const TextureDesc desc;
        const D3D12_RESOURCE_DESC resourceDesc;
        RefCountPtr<ID3D12Resource> resource;
        uint8_t planeCount = 1;
        HANDLE sharedHandle = nullptr;
        HeapHandle heap;

    // [rlaw] BEGIN
    #ifdef NVRHI_D3D12_WITH_D3D12MA
        D3D12MA::Allocation* m_Allocation = nullptr;
    #endif
    // [rlaw] END

        Texture(const Context& context, DeviceResources& resources, TextureDesc desc, const D3D12_RESOURCE_DESC& resourceDesc)
            : TextureStateExtension(this->desc)
            , desc(std::move(desc))
            , resourceDesc(resourceDesc)
            , m_Context(context)
            , m_Resources(resources)
        {
            TextureStateExtension::stateInitialized = true;
        }

        ~Texture() override;

        const TextureDesc& getDesc() const override { return desc; }

        Object getNativeObject(ObjectType objectType) override;
        Object getNativeView(ObjectType objectType, Format format, TextureSubresourceSet subresources, TextureDimension dimension, bool isReadOnlyDSV = false) override;

        void postCreate();
        void createSRV(size_t descriptor, Format format, TextureDimension dimension, TextureSubresourceSet subresources) const;
        void createUAV(size_t descriptor, Format format, TextureDimension dimension, TextureSubresourceSet subresources) const;
        void createRTV(size_t descriptor, Format format, TextureSubresourceSet subresources) const;
        void createDSV(size_t descriptor, TextureSubresourceSet subresources, bool isReadOnly = false) const;
        DescriptorIndex getClearMipLevelUAV(uint32_t mipLevel);

    private:
        const Context& m_Context;
        DeviceResources& m_Resources;

        TextureBindingKey_HashMap<DescriptorIndex> m_RenderTargetViews;
        TextureBindingKey_HashMap<DescriptorIndex> m_DepthStencilViews;
        TextureBindingKey_HashMap<DescriptorIndex> m_CustomSRVs;
        TextureBindingKey_HashMap<DescriptorIndex> m_CustomUAVs;
        std::vector<DescriptorIndex> m_ClearMipLevelUAVs;
    };

    class Buffer : public RefCounter<IBuffer>, public BufferStateExtension
    {
    public:
        const BufferDesc desc;
        RefCountPtr<ID3D12Resource> resource;
        D3D12_GPU_VIRTUAL_ADDRESS gpuVA{};
        D3D12_RESOURCE_DESC resourceDesc{};

        HeapHandle heap;

        RefCountPtr<ID3D12Fence> lastUseFence;
        uint64_t lastUseFenceValue = 0;
        HANDLE sharedHandle = nullptr;

    // [rlaw] BEGIN
    #ifdef NVRHI_D3D12_WITH_D3D12MA
        D3D12MA::Allocation* m_Allocation = nullptr;
    #endif
    // [rlaw] END

        Buffer(const Context& context, DeviceResources& resources, BufferDesc desc)
            : BufferStateExtension(this->desc)
            , desc(std::move(desc))
            , m_Context(context)
            , m_Resources(resources)
        { }

        ~Buffer() override;
        
        const BufferDesc& getDesc() const override { return desc; }
        GpuVirtualAddress getGpuVirtualAddress() const override { return gpuVA; }

        Object getNativeObject(ObjectType objectType) override;

        void postCreate();
        DescriptorIndex getClearUAV();
        void createCBV(size_t descriptor, BufferRange range) const;
        void createSRV(size_t descriptor, Format format, BufferRange range, ResourceType type) const;
        void createUAV(size_t descriptor, Format format, BufferRange range, ResourceType type) const;
        static void createNullSRV(size_t descriptor, Format format, const Context& context);
        static void createNullUAV(size_t descriptor, Format format, const Context& context);

    private:
        const Context& m_Context;
        DeviceResources& m_Resources;
        DescriptorIndex m_ClearUAV = c_InvalidDescriptorIndex;
    };

    class StagingTexture : public RefCounter<IStagingTexture>
    {
    public:
        TextureDesc desc;
        D3D12_RESOURCE_DESC resourceDesc{};
        RefCountPtr<Buffer> buffer;
        CpuAccessMode cpuAccess = CpuAccessMode::None;
        std::vector<UINT64> subresourceOffsets;

        RefCountPtr<ID3D12Fence> lastUseFence;
        uint64_t lastUseFenceValue = 0;

        struct SliceRegion
        {
            // offset and size in bytes of this region inside the buffer
            off_t offset = 0;
            size_t size = 0;

            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        };

        SliceRegion mappedRegion;
        CpuAccessMode mappedAccess = CpuAccessMode::None;

        // returns a SliceRegion struct corresponding to the subresource that slice points at
        // note that this always returns the entire subresource
        SliceRegion getSliceRegion(ID3D12Device *device, const TextureSlice& slice);

        // returns the total size in bytes required for this staging texture
        size_t getSizeInBytes(ID3D12Device *device);

        void computeSubresourceOffsets(ID3D12Device *device);
        
        const TextureDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class SamplerFeedbackTexture : public RefCounter<ISamplerFeedbackTexture>, public TextureStateExtension
    {
    public:
        const SamplerFeedbackTextureDesc desc;
        const TextureDesc textureDesc; // used with state tracking
        RefCountPtr<ID3D12Resource> resource;
        TextureHandle pairedTexture;
        DescriptorIndex clearDescriptorIndex = c_InvalidDescriptorIndex;

        SamplerFeedbackTexture(const Context& context, DeviceResources& resources, SamplerFeedbackTextureDesc desc, TextureDesc textureDesc, ITexture* pairedTexture)
            : desc(std::move(desc))
            , textureDesc(std::move(textureDesc))
            , m_Context(context)
            , m_Resources(resources)
            , pairedTexture(pairedTexture)
            , TextureStateExtension(SamplerFeedbackTexture::textureDesc)
        {
            TextureStateExtension::stateInitialized = true;
            TextureStateExtension::isSamplerFeedback = true;
        }

        // [rlaw] BEGIN
    #ifdef NVRHI_D3D12_WITH_D3D12MA
        ~SamplerFeedbackTexture() override;
    #endif // #ifdef NVRHI_D3D12_WITH_D3D12MA
        // [rlaw] END

        const SamplerFeedbackTextureDesc& getDesc() const override { return desc; }
        TextureHandle getPairedTexture() override { return pairedTexture; }

        void createUAV(size_t descriptor) const;

        Object getNativeObject(ObjectType objectType) override;

    // [rlaw] BEGIN
    #ifdef NVRHI_D3D12_WITH_D3D12MA
        D3D12MA::Allocation* m_Allocation = nullptr;
    #endif
    // [rlaw] END

    private:
        const Context& m_Context;
        DeviceResources& m_Resources;
    };

    class Sampler : public RefCounter<ISampler>
    {
    public:
        Sampler(const Context& context, const SamplerDesc& desc);
        
        void createDescriptor(size_t descriptor) const;

        const SamplerDesc& getDesc() const override { return m_Desc; }

    private:
        const Context& m_Context;
        const SamplerDesc m_Desc;
        D3D12_SAMPLER_DESC m_d3d12desc;
    };

    class InputLayout : public RefCounter<IInputLayout>
    {
    public:
        std::vector<VertexAttributeDesc> attributes;
        std::vector<D3D12_INPUT_ELEMENT_DESC> inputElements;

        // maps a binding slot to an element stride
        std::unordered_map<uint32_t, uint32_t> elementStrides;
        
        uint32_t getNumAttributes() const override;
        const VertexAttributeDesc* getAttributeDesc(uint32_t index) const override;
    };

    class EventQuery : public RefCounter<IEventQuery>
    {
    public:
        RefCountPtr<ID3D12Fence> fence;
        uint64_t fenceCounter = 0;
        bool started = false;
        bool resolved = false;
    };

    class TimerQuery : public RefCounter<ITimerQuery>
    {
    public:
        uint32_t beginQueryIndex = 0;
        uint32_t endQueryIndex = 0;

        RefCountPtr<ID3D12Fence> fence;
        uint64_t fenceCounter = 0;

        bool started = false;
        bool resolved = false;
        float time = 0.f;

        TimerQuery(DeviceResources& resources)
            : m_Resources(resources)
        { }

        ~TimerQuery() override;

    private:
        DeviceResources& m_Resources;
    };

    // [rlaw] BEGIN: Pipeline Query support
    class PipelineStatisticsQuery : public RefCounter<IPipelineStatisticsQuery>
    {
    public:
        uint32_t queryIndex = 0;

        RefCountPtr<ID3D12Fence> fence;
        uint64_t fenceCounter = 0;

        bool started = false;
        bool resolved = false;
        PipelineStatistics statistics;

        PipelineStatisticsQuery(DeviceResources& resources)
            : m_Resources(resources)
        {
        }

        ~PipelineStatisticsQuery() override;

    private:
        DeviceResources& m_Resources;
    };
    // [rlaw] END: Pipeline Query support

    class BindingLayout : public RefCounter<IBindingLayout>
    {
    public:
        BindingLayoutDesc desc;
        uint32_t pushConstantByteSize = 0;
        RootParameterIndex rootParameterPushConstants = ~0u;
        RootParameterIndex rootParameterSRVetc = ~0u;
        RootParameterIndex rootParameterSamplers = ~0u;
        int descriptorTableSizeSRVetc = 0;
        int descriptorTableSizeSamplers = 0;
        std::vector<D3D12_DESCRIPTOR_RANGE1> descriptorRangesSRVetc;
        std::vector<D3D12_DESCRIPTOR_RANGE1> descriptorRangesSamplers;
        std::vector<BindingLayoutItem> bindingLayoutsSRVetc;
        static_vector<std::pair<RootParameterIndex, D3D12_ROOT_DESCRIPTOR1>, c_MaxVolatileConstantBuffersPerLayout> rootParametersVolatileCB;
        static_vector<D3D12_ROOT_PARAMETER1, 32> rootParameters;

        BindingLayout(const BindingLayoutDesc& desc);

        const BindingLayoutDesc* getDesc() const override { return &desc; }
        const BindlessLayoutDesc* getBindlessDesc() const override { return nullptr; }
    };

    class BindlessLayout : public RefCounter<IBindingLayout>
    {
    public:
        BindlessLayoutDesc desc;
        static_vector<D3D12_DESCRIPTOR_RANGE1, 32> descriptorRanges;
        D3D12_ROOT_PARAMETER1 rootParameter{};

        BindlessLayout(const BindlessLayoutDesc& desc);

        const BindingLayoutDesc* getDesc() const override { return nullptr; }
        const BindlessLayoutDesc* getBindlessDesc() const override { return &desc; }
    };

    class RootSignature : public RefCounter<IRootSignature>
    {
    public:
        size_t hash = 0;
        static_vector<std::pair<BindingLayoutHandle, RootParameterIndex>, c_MaxBindingLayouts> pipelineLayouts;
        RefCountPtr<ID3D12RootSignature> handle;
        uint32_t pushConstantByteSize = 0;
        RootParameterIndex rootParameterPushConstants = ~0u;
        
        RootSignature(DeviceResources& resources)
            : m_Resources(resources)
        { }

        ~RootSignature() override;
        Object getNativeObject(ObjectType objectType) override;

    private:
        DeviceResources& m_Resources;
    };

    class Framebuffer : public RefCounter<IFramebuffer>
    {
    public:
        FramebufferDesc desc;
        FramebufferInfoEx framebufferInfo;

        static_vector<TextureHandle, c_MaxRenderTargets + 1> textures;
        static_vector<DescriptorIndex, c_MaxRenderTargets> RTVs;
        DescriptorIndex DSV = c_InvalidDescriptorIndex;
        uint32_t rtWidth = 0;
        uint32_t rtHeight = 0;

        Framebuffer(DeviceResources& resources)
            : m_Resources(resources)
        { }

        ~Framebuffer() override;

        const FramebufferDesc& getDesc() const override { return desc; }
        const FramebufferInfoEx& getFramebufferInfo() const override { return framebufferInfo; }

    private:
        DeviceResources& m_Resources;
    };

    struct DX12_ViewportState
    {
        UINT numViewports = 0;
        D3D12_VIEWPORT viewports[16] = {};
        UINT numScissorRects = 0;
        D3D12_RECT scissorRects[16] = {};
    };

    class GraphicsPipeline : public RefCounter<IGraphicsPipeline>
    {
    public:
        GraphicsPipelineDesc desc;
        FramebufferInfo framebufferInfo;

        RefCountPtr<RootSignature> rootSignature;
        RefCountPtr<ID3D12PipelineState> pipelineState;

        bool requiresBlendFactor = false;
        
        const GraphicsPipelineDesc& getDesc() const override { return desc; }
        const FramebufferInfo& getFramebufferInfo() const override { return framebufferInfo; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class ComputePipeline : public RefCounter<IComputePipeline>
    {
    public:
        ComputePipelineDesc desc;

        RefCountPtr<RootSignature> rootSignature;
        RefCountPtr<ID3D12PipelineState> pipelineState;
        
        const ComputePipelineDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
    };

    class MeshletPipeline : public RefCounter<IMeshletPipeline>
    {
    public:
        MeshletPipelineDesc desc;
        FramebufferInfo framebufferInfo;

        RefCountPtr<RootSignature> rootSignature;
        RefCountPtr<ID3D12PipelineState> pipelineState;

        DX12_ViewportState viewportState;

        bool requiresBlendFactor = false;
        
        const MeshletPipelineDesc& getDesc() const override { return desc; }
        const FramebufferInfo& getFramebufferInfo() const override { return framebufferInfo; }
        Object getNativeObject(ObjectType objectType) override;
    };
    
    class BindingSet : public RefCounter<IBindingSet>
    {
    public:
        RefCountPtr<BindingLayout> layout;
        BindingSetDesc desc;

        // ShaderType -> DescriptorIndex
        DescriptorIndex descriptorTableSRVetc = 0;
        DescriptorIndex descriptorTableSamplers = 0;
        RootParameterIndex rootParameterIndexSRVetc = 0;
        RootParameterIndex rootParameterIndexSamplers = 0;
        bool descriptorTableValidSRVetc = false;
        bool descriptorTableValidSamplers = false;
        bool hasUavBindings = false;

        static_vector<std::pair<RootParameterIndex, IBuffer*>, c_MaxVolatileConstantBuffersPerLayout> rootParametersVolatileCB;
        
        std::vector<RefCountPtr<IResource>> resources;

        std::vector<uint16_t> bindingsThatNeedTransitions;

        BindingSet(const Context& context, DeviceResources& resources)
            : m_Context(context)
            , m_Resources(resources)
        { }

        ~BindingSet() override;

        void createDescriptors();

        const BindingSetDesc* getDesc() const override { return &desc; }
        IBindingLayout* getLayout() const override { return layout; }

    private:
        const Context& m_Context;
        DeviceResources& m_Resources;
    };

    class DescriptorTable : public RefCounter<IDescriptorTable>
    {
    public:
        uint32_t capacity = 0;
        DescriptorIndex firstDescriptor = 0;

        DescriptorTable(DeviceResources& resources)
            : m_Resources(resources)
        { }

        ~DescriptorTable() override;

        const BindingSetDesc* getDesc() const override { return nullptr; }
        IBindingLayout* getLayout() const override { return nullptr; }
        uint32_t getCapacity() const override { return capacity; }
        uint32_t getFirstDescriptorIndexInHeap() const override { return firstDescriptor; }

    private:
        DeviceResources& m_Resources;
    };

    DX12_ViewportState convertViewportState(const RasterState& rasterState, const FramebufferInfoEx& framebufferInfo, const ViewportState& vpState);

    class TextureState
    {
    public:
        std::vector<OptionalResourceState> subresourceStates;
        bool enableUavBarriers = true;
        bool firstUavBarrierPlaced = false;
        bool permanentTransition = false;

        TextureState(uint32_t numSubresources)
        {
            subresourceStates.resize(numSubresources, c_ResourceStateUnknown);
        }
    };

    class BufferState
    {
    public:
        OptionalResourceState state = c_ResourceStateUnknown;
        bool enableUavBarriers = true;
        bool firstUavBarrierPlaced = false;
        D3D12_GPU_VIRTUAL_ADDRESS volatileData = 0;
        bool permanentTransition = false;
    };

    D3D12_RESOURCE_STATES convertResourceStates(ResourceStates stateBits);
    
    class BufferChunk
    {
    public:
        static const uint64_t c_sizeAlignment = 4096; // GPU page size

        RefCountPtr<ID3D12Resource> buffer;
        uint64_t version = 0;
        uint64_t bufferSize = 0;
        uint64_t writePointer = 0;
        void* cpuVA = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS gpuVA = 0;
        uint32_t identifier = 0;

    // [rlaw] BEGIN
    #ifdef NVRHI_D3D12_WITH_D3D12MA
        D3D12MA::Allocation* m_Allocation = nullptr;
    #endif
    // [rlaw] END

        ~BufferChunk();
    };

    class UploadManager
    {
    public:
        UploadManager(const Context& context, class Queue* pQueue, size_t defaultChunkSize, uint64_t memoryLimit, bool isScratchBuffer);

        bool suballocateBuffer(uint64_t size, ID3D12GraphicsCommandList* pCommandList, ID3D12Resource** pBuffer, size_t* pOffset, void** pCpuVA,
            D3D12_GPU_VIRTUAL_ADDRESS* pGpuVA, uint64_t currentVersion, uint32_t alignment = 256);

        void submitChunks(uint64_t currentVersion, uint64_t submittedVersion);

    private:
        const Context& m_Context;
        Queue* m_Queue;
        size_t m_DefaultChunkSize = 0;
        uint64_t m_MemoryLimit = 0;
        uint64_t m_AllocatedMemory = 0;
        bool m_IsScratchBuffer = false;

        std::list<std::shared_ptr<BufferChunk>> m_ChunkPool;
        std::shared_ptr<BufferChunk> m_CurrentChunk;

        [[nodiscard]] std::shared_ptr<BufferChunk> createChunk(size_t size) const;
    };

    class OpacityMicromap : public RefCounter<rt::IOpacityMicromap>
    {
    public:
        RefCountPtr<d3d12::Buffer> dataBuffer;
        rt::OpacityMicromapDesc desc;
        bool allowUpdate = false;
        bool compacted = false;

        OpacityMicromap()
        { }

        Object getNativeObject(ObjectType objectType) override;

        const rt::OpacityMicromapDesc& getDesc() const override { return desc; }
        bool isCompacted() const override { return compacted; }
        uint64_t getDeviceAddress() const override;
    };

    class AccelStruct : public RefCounter<rt::IAccelStruct>
    {
    public:
        RefCountPtr<d3d12::Buffer> dataBuffer;
        std::vector<rt::AccelStructHandle> bottomLevelASes;
        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> dxrInstances;
        rt::AccelStructDesc desc;
        bool allowUpdate = false;
        bool compacted = false;
        size_t rtxmuId = ~0ull;
#ifdef NVRHI_WITH_RTXMU
        D3D12_GPU_VIRTUAL_ADDRESS rtxmuGpuVA = 0;
#endif

        AccelStruct(const Context& context)
            : m_Context(context)
        { }

        ~AccelStruct() override;

        void createSRV(size_t descriptor) const;

        Object getNativeObject(ObjectType objectType) override;

        const rt::AccelStructDesc& getDesc() const override { return desc; }
        bool isCompacted() const override { return compacted; }
        uint64_t getDeviceAddress() const override;
        
    private:
        const Context& m_Context;
    };

    class RayTracingPipeline : public RefCounter<rt::IPipeline>
    {
    public:
        rt::PipelineDesc desc;

        std::unordered_map<IBindingLayout*, RootSignatureHandle> localRootSignatures;
        RefCountPtr<RootSignature> globalRootSignature;
        RefCountPtr<ID3D12StateObject> pipelineState;
        RefCountPtr<ID3D12StateObjectProperties> pipelineInfo;

        struct ExportTableEntry
        {
            IBindingLayout* bindingLayout;
            const void* pShaderIdentifier;
        };

        std::unordered_map<std::string, ExportTableEntry> exports;
        uint32_t maxLocalRootParameters = 0;

        RayTracingPipeline(const Context& context)
            : m_Context(context)
        { }

        const ExportTableEntry* getExport(const char* name);
        uint32_t getShaderTableEntrySize() const;

        const rt::PipelineDesc& getDesc() const override { return desc; }
        rt::ShaderTableHandle createShaderTable() override;

    private:
        const Context& m_Context;
    };

    class ShaderTable : public RefCounter<rt::IShaderTable>
    {
    public:
        struct Entry
        {
            const void* pShaderIdentifier;
            BindingSetHandle localBindings;
        };

        RefCountPtr<RayTracingPipeline> pipeline;

        Entry rayGenerationShader = {};
        std::vector<Entry> missShaders;
        std::vector<Entry> callableShaders;
        std::vector<Entry> hitGroups;

        uint32_t version = 0;

        ShaderTable(const Context& context, RayTracingPipeline* _pipeline)
            : pipeline(_pipeline)
            , m_Context(context)
        { }

        uint32_t getNumEntries() const;

        void setRayGenerationShader(const char* exportName, IBindingSet* bindings = nullptr) override;
        int addMissShader(const char* exportName, IBindingSet* bindings = nullptr) override;
        int addHitGroup(const char* exportName, IBindingSet* bindings = nullptr) override;
        int addCallableShader(const char* exportName, IBindingSet* bindings = nullptr) override;
        void clearMissShaders() override;
        void clearHitShaders() override;
        void clearCallableShaders() override;
        rt::IPipeline* getPipeline() override;

    private:
        const Context& m_Context;

        bool verifyExport(const RayTracingPipeline::ExportTableEntry* pExport, IBindingSet* bindings) const;
    };


    class ShaderTableState
    {
    public:
        uint32_t committedVersion = 0;
        ID3D12DescriptorHeap* descriptorHeapSRV = nullptr;
        ID3D12DescriptorHeap* descriptorHeapSamplers = nullptr;
        D3D12_DISPATCH_RAYS_DESC dispatchRaysTemplate = {};
    };

    class Queue
    {
    public:
        RefCountPtr<ID3D12CommandQueue> queue;
        RefCountPtr<ID3D12Fence> fence;
        uint64_t lastSubmittedInstance = 0;
        uint64_t lastCompletedInstance = 0;
        std::atomic<uint64_t> recordingInstance = 1;
        std::deque<std::shared_ptr<class CommandListInstance>> commandListsInFlight;

        explicit Queue(const Context& context, ID3D12CommandQueue* queue);
        uint64_t updateLastCompletedInstance();

    private:
        const Context& m_Context;
    };
    
    class InternalCommandList
    {
    public:
        RefCountPtr<ID3D12CommandAllocator> allocator;
        RefCountPtr<ID3D12GraphicsCommandList> commandList;
        RefCountPtr<ID3D12GraphicsCommandList4> commandList4;
        RefCountPtr<ID3D12GraphicsCommandList6> commandList6;
#if NVRHI_D3D12_WITH_COOPVEC
        RefCountPtr<ID3D12GraphicsCommandListPreview> commandListPreview;
#endif
        uint64_t lastSubmittedInstance = 0;
#if NVRHI_WITH_AFTERMATH
        GFSDK_Aftermath_ContextHandle aftermathContext;
#endif
    };

    class CommandListInstance
    {
    public:
        uint64_t submittedInstance = 0;
        CommandQueue commandQueue = CommandQueue::Graphics;
        RefCountPtr<ID3D12Fence> fence;
        RefCountPtr<ID3D12CommandAllocator> commandAllocator;
        RefCountPtr<ID3D12CommandList> commandList;
        std::vector<RefCountPtr<IResource>> referencedResources;
        std::vector<RefCountPtr<IUnknown>> referencedNativeResources;
        std::vector<RefCountPtr<StagingTexture>> referencedStagingTextures;
        std::vector<RefCountPtr<Buffer>> referencedStagingBuffers;
        std::vector<RefCountPtr<TimerQuery>> referencedTimerQueries;
        std::vector<RefCountPtr<PipelineStatisticsQuery>> referencedPipelineStatisticsQueries; // [rlaw]: Pipeline Query support
#ifdef NVRHI_WITH_RTXMU
        std::vector<uint64_t> rtxmuBuildIds;
        std::vector<uint64_t> rtxmuCompactionIds;
#endif
    };

    class CommandList final : public RefCounter<nvrhi::d3d12::ICommandList>
    {
    public:

        // Internal interface functions

        CommandList(class Device* device, const Context& context, DeviceResources& resources, const CommandListParameters& params);
        ~CommandList() override;
        std::shared_ptr<CommandListInstance> executed(Queue* pQueue);
        void requireTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates state);
        void requireSamplerFeedbackTextureState(ISamplerFeedbackTexture* texture, ResourceStates state);
        void requireBufferState(IBuffer* buffer, ResourceStates state);
        ID3D12CommandList* getD3D12CommandList() const { return m_ActiveCommandList->commandList; }

        // IResource implementation

        Object getNativeObject(ObjectType objectType) override;

        // ICommandList implementation

        void open() override;
        void close() override;
        void clearState() override;
        
        void clearTextureFloat(ITexture* t, TextureSubresourceSet subresources, const Color& clearColor) override;
        void clearDepthStencilTexture(ITexture* t, TextureSubresourceSet subresources, bool clearDepth, float depth, bool clearStencil, uint8_t stencil) override;
        void clearTextureUInt(ITexture* t, TextureSubresourceSet subresources, uint32_t clearColor) override;
        void clearSamplerFeedbackTexture(ISamplerFeedbackTexture* texture) override;
        void decodeSamplerFeedbackTexture(IBuffer* buffer, ISamplerFeedbackTexture* texture, Format format) override;
        void setSamplerFeedbackTextureState(ISamplerFeedbackTexture* texture, ResourceStates stateBits) override;

        void copyTexture(ITexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice) override;
        void copyTexture(IStagingTexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice) override;
        void copyTexture(ITexture* dest, const TextureSlice& destSlice, IStagingTexture* src, const TextureSlice& srcSlice) override;
        void writeTexture(ITexture* dest, uint32_t arraySlice, uint32_t mipLevel, const void* data, size_t rowPitch, size_t depthPitch) override;
        void resolveTexture(ITexture* dest, const TextureSubresourceSet& dstSubresources, ITexture* src, const TextureSubresourceSet& srcSubresources) override;

        void writeBuffer(IBuffer* b, const void* data, size_t dataSize, uint64_t destOffsetBytes = 0) override;
        void clearBufferUInt(IBuffer* b, uint32_t clearValue) override;
        void copyBuffer(IBuffer* dest, uint64_t destOffsetBytes, IBuffer* src, uint64_t srcOffsetBytes, uint64_t dataSizeBytes) override;

        void setPushConstants(const void* data, size_t byteSize) override;

        void setGraphicsState(const GraphicsState& state) override;
        void draw(const DrawArguments& args) override;
        void drawIndexed(const DrawArguments& args) override;
        void drawIndirect(uint32_t offsetBytes, uint32_t drawCount) override;
        void drawIndexedIndirect(uint32_t offsetBytes, uint32_t drawCount) override;

        void setComputeState(const ComputeState& state) override;
        void dispatch(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1) override;
        void dispatchIndirect(uint32_t offsetBytes) override;

        void setMeshletState(const MeshletState& state) override;
        void dispatchMesh(uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1) override;
        void dispatchMeshIndirect(uint32_t offsetBytes) override; // [rlaw]: support dispatchMeshIndirect

        void setRayTracingState(const rt::State& state) override;
        void dispatchRays(const rt::DispatchRaysArguments& args) override;

        void buildOpacityMicromap(rt::IOpacityMicromap* omm, const rt::OpacityMicromapDesc& desc) override;
        void buildBottomLevelAccelStruct(rt::IAccelStruct* as, const rt::GeometryDesc* pGeometries, size_t numGeometries, rt::AccelStructBuildFlags buildFlags) override;
        void compactBottomLevelAccelStructs() override;
        void buildTopLevelAccelStruct(rt::IAccelStruct* as, const rt::InstanceDesc* pInstances, size_t numInstances, rt::AccelStructBuildFlags buildFlags) override;
        void buildTopLevelAccelStructFromBuffer(rt::IAccelStruct* as, nvrhi::IBuffer* instanceBuffer, uint64_t instanceBufferOffset, size_t numInstances,
            rt::AccelStructBuildFlags buildFlags = rt::AccelStructBuildFlags::None) override;
        void executeMultiIndirectClusterOperation(const rt::cluster::OperationDesc& desc) override;

        void convertCoopVecMatrices(coopvec::ConvertMatrixLayoutDesc const* convertDescs, size_t numDescs) override;

        void beginTimerQuery(ITimerQuery* query) override;
        void endTimerQuery(ITimerQuery* query) override;

        void beginMarker(const char *name) override;
        void endMarker() override;

        void setEnableAutomaticBarriers(bool enable) override;
        void setResourceStatesForBindingSet(IBindingSet* bindingSet) override;

        void setEnableUavBarriersForTexture(ITexture* texture, bool enableBarriers) override;
        void setEnableUavBarriersForBuffer(IBuffer* buffer, bool enableBarriers) override;

        void beginTrackingTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits) override;
        void beginTrackingBufferState(IBuffer* buffer, ResourceStates stateBits) override;

        void setTextureState(ITexture* texture, TextureSubresourceSet subresources, ResourceStates stateBits) override;
        void setBufferState(IBuffer* buffer, ResourceStates stateBits) override;
        void setAccelStructState(rt::IAccelStruct* as, ResourceStates stateBits) override;
        
        void setPermanentTextureState(ITexture* texture, ResourceStates stateBits) override;
        void setPermanentBufferState(IBuffer* buffer, ResourceStates stateBits) override;

        void commitBarriers() override;

        ResourceStates getTextureSubresourceState(ITexture* texture, ArraySlice arraySlice, MipLevel mipLevel) override;
        ResourceStates getBufferState(IBuffer* buffer) override;

        nvrhi::IDevice* getDevice() override;
        const CommandListParameters& getDesc() override { return m_Desc; }

        // D3D12 specific methods

        bool allocateUploadBuffer(size_t size, void** pCpuAddress, D3D12_GPU_VIRTUAL_ADDRESS* pGpuAddress) override;
        bool allocateDxrScratchBuffer(size_t size, void** pCpuAddress, D3D12_GPU_VIRTUAL_ADDRESS* pGpuAddress);
        bool commitDescriptorHeaps() override;
        D3D12_GPU_VIRTUAL_ADDRESS getBufferGpuVA(IBuffer* buffer) override;

        void updateGraphicsVolatileBuffers() override;
        void updateComputeVolatileBuffers() override;

        void setComputeBindings(const BindingSetVector& bindings, uint32_t bindingUpdateMask, IBuffer* indirectParams, bool updateIndirectParams, const RootSignature* rootSignature);
        void setGraphicsBindings(const BindingSetVector& bindings, uint32_t bindingUpdateMask, IBuffer* indirectParams, bool updateIndirectParams, const RootSignature* rootSignature);

        // [rlaw] BEGIN: Pipeline Query support
        void beginPipelineStatisticsQuery(IPipelineStatisticsQuery* query) override;
        void endPipelineStatisticsQuery(IPipelineStatisticsQuery* query) override;
        // [rlaw] END: Pipeline Query support
        
    private:
        const Context& m_Context;
        DeviceResources& m_Resources;
        
        struct VolatileConstantBufferBinding
        {
            uint32_t bindingPoint; // RootParameterIndex
            Buffer* buffer;
            D3D12_GPU_VIRTUAL_ADDRESS address;
        };
        
        IDevice* m_Device;
        Queue* m_Queue;
        UploadManager m_UploadManager;
        UploadManager m_DxrScratchManager;
        CommandListResourceStateTracker m_StateTracker;
        bool m_EnableAutomaticBarriers = true;
        
        CommandListParameters m_Desc;

        std::shared_ptr<InternalCommandList> m_ActiveCommandList;
        std::list<std::shared_ptr<InternalCommandList>> m_CommandListPool;
        std::shared_ptr<CommandListInstance> m_Instance;
        uint64_t m_RecordingVersion = 0;
#if NVRHI_WITH_AFTERMATH
        AftermathMarkerTracker m_AftermathTracker;
#endif

        // Cache for user-provided state

        GraphicsState m_CurrentGraphicsState;
        ComputeState m_CurrentComputeState;
        MeshletState m_CurrentMeshletState;
        rt::State m_CurrentRayTracingState;
        bool m_CurrentGraphicsStateValid = false;
        bool m_CurrentComputeStateValid = false;
        bool m_CurrentMeshletStateValid = false;
        bool m_CurrentRayTracingStateValid = false;

        // Cache for internal state

        ID3D12DescriptorHeap* m_CurrentHeapSRVetc = nullptr;
        ID3D12DescriptorHeap* m_CurrentHeapSamplers = nullptr;
        ID3D12Resource* m_CurrentUploadBuffer = nullptr;
        SinglePassStereoState m_CurrentSinglePassStereoState;
        
        std::unordered_map<IBuffer*, D3D12_GPU_VIRTUAL_ADDRESS> m_VolatileConstantBufferAddresses;
        bool m_AnyVolatileBufferWrites = false;

        std::vector<D3D12_RESOURCE_BARRIER> m_D3DBarriers; // Used locally in commitBarriers, member to avoid re-allocations

        // Bound volatile buffer state. Saves currently bound volatile buffers and their current GPU VAs.
        // Necessary to patch the bound VAs when a buffer is updated between setGraphicsState and draw, or between draws.

        static_vector<VolatileConstantBufferBinding, c_MaxVolatileConstantBuffers> m_CurrentGraphicsVolatileCBs;
        static_vector<VolatileConstantBufferBinding, c_MaxVolatileConstantBuffers> m_CurrentComputeVolatileCBs;

        std::unordered_map<rt::IShaderTable*, std::unique_ptr<ShaderTableState>> m_ShaderTableStates;
        ShaderTableState* getShaderTableStateTracking(rt::IShaderTable* shaderTable);
        
        void clearStateCache();

        void bindGraphicsPipeline(GraphicsPipeline* pso, bool updateRootSignature) const;
        void bindMeshletPipeline(MeshletPipeline* pso, bool updateRootSignature) const;
        void bindFramebuffer(Framebuffer* fb);
        void unbindShadingRateState();
        
        std::shared_ptr<InternalCommandList> createInternalCommandList() const;

        void buildTopLevelAccelStructInternal(AccelStruct* as, D3D12_GPU_VIRTUAL_ADDRESS instanceData, size_t numInstances, rt::AccelStructBuildFlags buildFlags);
    };

    class Device final : public RefCounter<IDevice>
    {
    public:
        explicit Device(const DeviceDesc& desc);
        ~Device() override;
        
        // IResource implementation
        
        Object getNativeObject(ObjectType objectType) override;

        // IDevice implementation

        HeapHandle createHeap(const HeapDesc& d) override;

        TextureHandle createTexture(const TextureDesc& d) override;
        MemoryRequirements getTextureMemoryRequirements(ITexture* texture) override;
        bool bindTextureMemory(ITexture* texture, IHeap* heap, uint64_t offset) override;

        TextureHandle createHandleForNativeTexture(ObjectType objectType, Object texture, const TextureDesc& desc) override;

        StagingTextureHandle createStagingTexture(const TextureDesc& d, CpuAccessMode cpuAccess) override;
        void *mapStagingTexture(IStagingTexture* tex, const TextureSlice& slice, CpuAccessMode cpuAccess, size_t *outRowPitch) override;
        void unmapStagingTexture(IStagingTexture* tex) override;

        void getTextureTiling(ITexture* texture, uint32_t* numTiles, PackedMipDesc* desc, TileShape* tileShape, uint32_t* subresourceTilingsNum, SubresourceTiling* subresourceTilings) override;
        void updateTextureTileMappings(ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings, CommandQueue executionQueue = CommandQueue::Graphics) override;

        SamplerFeedbackTextureHandle createSamplerFeedbackTexture(ITexture* pairedTexture, const SamplerFeedbackTextureDesc& desc) override;
        SamplerFeedbackTextureHandle createSamplerFeedbackForNativeTexture(ObjectType objectType, Object texture, ITexture* pairedTexture) override;

        BufferHandle createBuffer(const BufferDesc& d) override;
        void *mapBuffer(IBuffer* b, CpuAccessMode mapFlags) override;
        void unmapBuffer(IBuffer* b) override;
        MemoryRequirements getBufferMemoryRequirements(IBuffer* buffer) override;
        bool bindBufferMemory(IBuffer* buffer, IHeap* heap, uint64_t offset) override;

        BufferHandle createHandleForNativeBuffer(ObjectType objectType, Object buffer, const BufferDesc& desc) override;

        ShaderHandle createShader(const ShaderDesc& d, const void* binary, size_t binarySize) override;
        ShaderHandle createShaderSpecialization(IShader* baseShader, const ShaderSpecialization* constants, uint32_t numConstants) override;
        ShaderLibraryHandle createShaderLibrary(const void* binary, size_t binarySize) override;

        SamplerHandle createSampler(const SamplerDesc& d) override;

        InputLayoutHandle createInputLayout(const VertexAttributeDesc* d, uint32_t attributeCount, IShader* vertexShader) override;

        EventQueryHandle createEventQuery() override;
        void setEventQuery(IEventQuery* query, CommandQueue queue) override;
        bool pollEventQuery(IEventQuery* query) override;
        void waitEventQuery(IEventQuery* query) override;
        void resetEventQuery(IEventQuery* query) override;

        TimerQueryHandle createTimerQuery() override;
        bool pollTimerQuery(ITimerQuery* query) override;
        float getTimerQueryTime(ITimerQuery* query) override;
        void resetTimerQuery(ITimerQuery* query) override;

        // [rlaw] BEGIN: Pipeline Query support
        PipelineStatisticsQueryHandle createPipelineStatisticsQuery() override;
        PipelineStatistics getPipelineStatistics(IPipelineStatisticsQuery* query) override;
        bool pollPipelineStatisticsQuery(IPipelineStatisticsQuery* query) override;
        void resetPipelineStatisticsQuery(IPipelineStatisticsQuery* query) override;
        // [rlaw] END: Pipeline Query support

        GraphicsAPI getGraphicsAPI() override;

        FramebufferHandle createFramebuffer(const FramebufferDesc& desc) override;
        
        GraphicsPipelineHandle createGraphicsPipeline(const GraphicsPipelineDesc& desc, IFramebuffer* fb) override;
        
        ComputePipelineHandle createComputePipeline(const ComputePipelineDesc& desc) override;

        MeshletPipelineHandle createMeshletPipeline(const MeshletPipelineDesc& desc, IFramebuffer* fb) override;

        rt::PipelineHandle createRayTracingPipeline(const rt::PipelineDesc& desc) override;

        BindingLayoutHandle createBindingLayout(const BindingLayoutDesc& desc) override;
        BindingLayoutHandle createBindlessLayout(const BindlessLayoutDesc& desc) override;

        BindingSetHandle createBindingSet(const BindingSetDesc& desc, IBindingLayout* layout) override;
        DescriptorTableHandle createDescriptorTable(IBindingLayout* layout) override;

        void resizeDescriptorTable(IDescriptorTable* descriptorTable, uint32_t newSize, bool keepContents = true) override;
        bool writeDescriptorTable(IDescriptorTable* descriptorTable, const BindingSetItem& item) override;

        rt::OpacityMicromapHandle createOpacityMicromap(const rt::OpacityMicromapDesc& desc) override;
        rt::AccelStructHandle createAccelStruct(const rt::AccelStructDesc& desc) override;
        MemoryRequirements getAccelStructMemoryRequirements(rt::IAccelStruct* as) override;
        rt::cluster::OperationSizeInfo getClusterOperationSizeInfo(const rt::cluster::OperationParams& params) override;

        bool bindAccelStructMemory(rt::IAccelStruct* as, IHeap* heap, uint64_t offset) override;

        nvrhi::CommandListHandle createCommandList(const CommandListParameters& params = CommandListParameters()) override;
        uint64_t executeCommandLists(nvrhi::ICommandList* const* pCommandLists, size_t numCommandLists, CommandQueue executionQueue = CommandQueue::Graphics) override;
        void queueWaitForCommandList(CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance) override;
        bool waitForIdle() override;
        void runGarbageCollection() override;
        bool queryFeatureSupport(Feature feature, void* pInfo = nullptr, size_t infoSize = 0) override;
        FormatSupport queryFormatSupport(Format format) override;
        coopvec::DeviceFeatures queryCoopVecFeatures() override;
        size_t getCoopVecMatrixSize(coopvec::DataType type, coopvec::MatrixLayout layout, int rows, int columns) override;
        Object getNativeQueue(ObjectType objectType, CommandQueue queue) override;
        IMessageCallback* getMessageCallback() override { return m_Context.messageCallback; }
        bool isAftermathEnabled() override { return m_AftermathEnabled; }
        AftermathCrashDumpHelper& getAftermathCrashDumpHelper() override { return m_AftermathCrashDumpHelper; }

        // d3d12::IDevice implementation

        RootSignatureHandle buildRootSignature(const static_vector<BindingLayoutHandle, c_MaxBindingLayouts>& pipelineLayouts, bool allowInputLayout, bool isLocal, const D3D12_ROOT_PARAMETER1* pCustomParameters = nullptr, uint32_t numCustomParameters = 0) override;
        GraphicsPipelineHandle createHandleForNativeGraphicsPipeline(IRootSignature* rootSignature, ID3D12PipelineState* pipelineState, const GraphicsPipelineDesc& desc, const FramebufferInfo& framebufferInfo) override;
        MeshletPipelineHandle createHandleForNativeMeshletPipeline(IRootSignature* rootSignature, ID3D12PipelineState* pipelineState, const MeshletPipelineDesc& desc, const FramebufferInfo& framebufferInfo) override;
        IDescriptorHeap* getDescriptorHeap(DescriptorHeapType heapType) override;

        // Internal interface
        Queue* getQueue(CommandQueue type) { return m_Queues[int(type)].get(); }

        Context& getContext() { return m_Context; }

        bool setHlslExtensionsUAV(uint32_t slot);

        bool GetAccelStructPreBuildInfo(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& outPreBuildInfo, const rt::AccelStructDesc& desc) const;

        bool GetNvapiIsInitialized() const { return m_NvapiIsInitialized; }
        bool GetOpacityMicromapSupported() const { return m_OpacityMicromapSupported; }
        bool GetLinearSweptSpheresSupported( ) const { return m_LinearSweptSpheresSupported; }

    private:
        Context m_Context;
        DeviceResources m_Resources;

        std::array<std::unique_ptr<Queue>, (int)CommandQueue::Count> m_Queues;
        HANDLE m_FenceEvent;

        std::mutex m_Mutex;

        std::vector<ID3D12CommandList*> m_CommandListsToExecute; // used locally in executeCommandLists, member to avoid re-allocations
        
        bool m_NvapiIsInitialized = false;
        bool m_SinglePassStereoSupported = false;
        bool m_HlslExtensionsSupported = false;
        bool m_FastGeometryShaderSupported = false;
        bool m_RayTracingSupported = false;
        bool m_TraceRayInlineSupported = false;
        bool m_MeshletsSupported = false;
        bool m_VariableRateShadingSupported = false;
        bool m_OpacityMicromapSupported = false;
        bool m_RayTracingClustersSupported = false;
        bool m_LinearSweptSpheresSupported = false;
        bool m_SpheresSupported = false;
        bool m_ShaderExecutionReorderingSupported = false;
        bool m_SamplerFeedbackSupported = false;
        bool m_AftermathEnabled = false;
        bool m_HeapDirectlyIndexedEnabled = false;
        bool m_CoopVecInferencingSupported = false;
        bool m_CoopVecTrainingSupported = false;
        AftermathCrashDumpHelper m_AftermathCrashDumpHelper;


        D3D12_FEATURE_DATA_D3D12_OPTIONS  m_Options = {};
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 m_Options1 = {};
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 m_Options5 = {};
        D3D12_FEATURE_DATA_D3D12_OPTIONS6 m_Options6 = {};
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 m_Options7 = {};

        RefCountPtr<RootSignature> getRootSignature(const static_vector<BindingLayoutHandle, c_MaxBindingLayouts>& pipelineLayouts, bool allowInputLayout);
        RefCountPtr<ID3D12PipelineState> createPipelineState(const GraphicsPipelineDesc& desc, RootSignature* pRS, const FramebufferInfo& fbinfo) const;
        RefCountPtr<ID3D12PipelineState> createPipelineState(const ComputePipelineDesc& desc, RootSignature* pRS) const;
        RefCountPtr<ID3D12PipelineState> createPipelineState(const MeshletPipelineDesc& desc, RootSignature* pRS, const FramebufferInfo& fbinfo) const;
    
        // [rlaw] BEGIN
    #ifdef NVRHI_D3D12_WITH_D3D12MA
        D3D12MA::Allocator* m_Allocator = nullptr;
    #endif
        // [rlaw] END
    };

} // namespace nvrhi::d3d12
