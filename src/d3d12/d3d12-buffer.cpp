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


#include "d3d12-backend.h"
#include <nvrhi/utils.h>
#include <nvrhi/common/misc.h>

#include <sstream>
#include <iomanip>

namespace nvrhi::d3d12
{
    Object Buffer::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        case ObjectTypes::D3D12_Resource:
            return Object(resource);
        case ObjectTypes::SharedHandle:
            return Object(sharedHandle);
        default:
            return nullptr;
        }
    }
    
    Buffer::~Buffer()
    {
        if (m_Context.logBufferLifetime)
        {
            std::stringstream ss;
            ss << "Release buffer: " << desc.debugName << " 0x" << std::hex << getGpuVirtualAddress();
            m_Context.info(ss.str());
        }

        if (m_ClearUAV != c_InvalidDescriptorIndex)
        {
            m_Resources.shaderResourceViewHeap.releaseDescriptor(m_ClearUAV);
            m_ClearUAV = c_InvalidDescriptorIndex;
        }

    // [rlaw] BEGIN
    #ifdef NVRHI_D3D12_WITH_D3D12MA
        if (m_Allocation)
        {
            m_Allocation->Release();
            m_Allocation = nullptr;
        }
    #endif // #ifdef NVRHI_D3D12_WITH_D3D12MA
    // [rlaw] END
    }

    BufferHandle Device::createBuffer(const BufferDesc& d)
    {
        BufferDesc desc = d;
        if (desc.isConstantBuffer)
        {
            desc.byteSize = align(d.byteSize, 256ull);
        }

        Buffer* buffer = new Buffer(m_Context, m_Resources, desc);
        
        if (d.isVolatile)
        {
            // Do not create any resources for volatile buffers. Done.
            return BufferHandle::Create(buffer);
        }

        D3D12_RESOURCE_DESC& resourceDesc = buffer->resourceDesc;
        resourceDesc.Width = buffer->desc.byteSize;
        resourceDesc.Height = 1;
        resourceDesc.DepthOrArraySize = 1;
        resourceDesc.MipLevels = 1;
        resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
        resourceDesc.SampleDesc.Count = 1;
        resourceDesc.SampleDesc.Quality = 0;
        resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        if (buffer->desc.canHaveUAVs)
            resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        if (d.isVirtual)
        {
            return BufferHandle::Create(buffer);
        }

        D3D12_HEAP_PROPERTIES heapProps = {};
        D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE;
        D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;

        bool isShared = false;
        if ((d.sharedResourceFlags & SharedResourceFlags::Shared) != 0) {
            heapFlags |= D3D12_HEAP_FLAG_SHARED;
            isShared = true;
        }
        if ((d.sharedResourceFlags & SharedResourceFlags::Shared_CrossAdapter) != 0) {
            resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
            heapFlags |= D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;
            isShared = true;
        }

        switch(buffer->desc.cpuAccess)
        {
            case CpuAccessMode::None:
                heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
                initialState = convertResourceStates(d.initialState);
                if (initialState != D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)
                    initialState = D3D12_RESOURCE_STATE_COMMON;
                break;

            case CpuAccessMode::Read:
                heapProps.Type = D3D12_HEAP_TYPE_READBACK;
                initialState = D3D12_RESOURCE_STATE_COPY_DEST;
                break;

            case CpuAccessMode::Write:
                heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
                initialState = D3D12_RESOURCE_STATE_GENERIC_READ;
                break;
        }

        // Allow readback buffers to be used as resolve destination targets
        if ((buffer->desc.cpuAccess == CpuAccessMode::Read) && (d.initialState == ResourceStates::ResolveDest))
        {
            heapProps.Type = D3D12_HEAP_TYPE_CUSTOM;
            heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
            heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
            initialState = D3D12_RESOURCE_STATE_COMMON;
        }

    // [rlaw]: D3D12MA
    #ifdef NVRHI_D3D12_WITH_D3D12MA

        HRESULT res;

        const bool bIsResolveReadbackBuffer = ((buffer->desc.cpuAccess == CpuAccessMode::Read) && (d.initialState == ResourceStates::ResolveDest));
        if (!bIsResolveReadbackBuffer)
        {
            D3D12MA::ALLOCATION_DESC allocDesc{};
            allocDesc.Flags = D3D12MA::ALLOCATION_FLAG_WITHIN_BUDGET;
            allocDesc.HeapType = heapProps.Type;
            allocDesc.ExtraHeapFlags = heapFlags;

            assert(m_Allocator);
            res = m_Allocator->CreateResource(
                &allocDesc,
                &resourceDesc,
                convertResourceStates(d.initialState),
                nullptr,
                &buffer->m_Allocation,
                IID_PPV_ARGS(&buffer->resource));
        }
        else
        {
            res = m_Context.device->CreateCommittedResource(
                &heapProps,
                heapFlags,
                &resourceDesc,
                initialState,
                nullptr,
                IID_PPV_ARGS(&buffer->resource));
        }

    #else // NVRHI_D3D12_WITH_D3D12MA

        HRESULT res = m_Context.device->CreateCommittedResource(
            &heapProps,
            heapFlags,
            &resourceDesc,
            initialState,
            nullptr,
            IID_PPV_ARGS(&buffer->resource));

    #endif // NVRHI_D3D12_WITH_D3D12MA

        if (FAILED(res))
        {
            std::stringstream ss;
            ss << "CreateCommittedResource call failed for buffer " << utils::DebugNameToString(d.debugName)
                << ", HRESULT = 0x" << std::hex << std::setw(8) << res;
            m_Context.error(ss.str());

            delete buffer;
            return nullptr;
        }
        
        if (isShared)
        {
            res = m_Context.device->CreateSharedHandle(
                buffer->resource,
                nullptr,
                GENERIC_ALL,
                nullptr,
                &buffer->sharedHandle);

            if (FAILED(res))
            {
                std::stringstream ss;
                ss << "Failed to create shared handle " << utils::DebugNameToString(d.debugName) << ", error code = 0x";
                ss.setf(std::ios::hex, std::ios::basefield);
                ss << res;
                m_Context.error(ss.str());

                delete buffer;
                return nullptr;
            }
        }

        buffer->postCreate();

        return BufferHandle::Create(buffer);
    }

    void Buffer::postCreate()
    {
        gpuVA = resource->GetGPUVirtualAddress();

        if (!desc.debugName.empty())
        {
            std::wstring wname(desc.debugName.begin(), desc.debugName.end());
            resource->SetName(wname.c_str());
#if NVRHI_WITH_AFTERMATH
            // the driver will track the resource internally so don't need to keep the handle around
            GFSDK_Aftermath_ResourceHandle resourceHandle = {};
            GFSDK_Aftermath_DX12_RegisterResource(resource, &resourceHandle);
#endif

        // [rlaw] BEGIN
        #ifdef NVRHI_D3D12_WITH_D3D12MA
            if (m_Allocation)
            {
                m_Allocation->SetName(wname.c_str());
            }
        #endif
        // [rlaw] END

        }

        if (m_Context.logBufferLifetime)
        {
            size_t byteDisplay = desc.byteSize;
            const char* byteUnit = "B";

            if (desc.byteSize > (1 << 20))
            {
                byteDisplay = desc.byteSize >> 20;
                byteUnit = "MB";
            }
            else if (desc.byteSize > (1 << 10))
            {
                byteDisplay = desc.byteSize >> 10;
                byteUnit = "KB";
            }

            std::stringstream ss;
            ss << "Create buffer: " << desc.debugName 
               << " Res:0x" << std::hex << reinterpret_cast<uintptr_t>(resource.Get()) 
               << " Gpu:0x" << std::hex << getGpuVirtualAddress() << "->0x" << std::hex << getGpuVirtualAddress() + desc.byteSize;

            if (desc.structStride)
            {
                ss << " (n:" << std::dec << (desc.structStride ? desc.byteSize / desc.structStride : 0) 
                   << " stride:" << std::dec << desc.structStride
                   << "B size:" << std::dec << byteDisplay << byteUnit << ")";
            }
            else
            {
                ss << " (size:" << std::dec << byteDisplay << byteUnit << ")";
            }
    
            m_Context.info(ss.str());
        }
    }

    DescriptorIndex Buffer::getClearUAV()
    {
        assert(desc.canHaveUAVs);

        if (m_ClearUAV != c_InvalidDescriptorIndex)
            return m_ClearUAV;

        m_ClearUAV = m_Resources.shaderResourceViewHeap.allocateDescriptor();
        createUAV(m_Resources.shaderResourceViewHeap.getCpuHandle(m_ClearUAV).ptr, Format::R32_UINT,
            EntireBuffer, ResourceType::TypedBuffer_UAV);
        m_Resources.shaderResourceViewHeap.copyToShaderVisibleHeap(m_ClearUAV);
        return m_ClearUAV;
    }

    void *Device::mapBuffer(IBuffer* _b, CpuAccessMode flags)
    {
        Buffer* b = checked_cast<Buffer*>(_b);

        if (b->lastUseFence)
        {
            WaitForFence(b->lastUseFence, b->lastUseFenceValue, m_FenceEvent);
            b->lastUseFence = nullptr;
        }

        D3D12_RANGE range;

        if (flags == CpuAccessMode::Read)
        {
            range = { 0, b->desc.byteSize };
        } else {
            range = { 0, 0 };
        }

        void *mappedBuffer;
        const HRESULT res = b->resource->Map(0, &range, &mappedBuffer);

        if (FAILED(res))
        {
            std::stringstream ss;
            ss << "Map call failed for buffer " << utils::DebugNameToString(b->desc.debugName)
               << ", HRESULT = 0x" << std::hex << std::setw(8) << res;
            m_Context.error(ss.str());
            
            return nullptr;
        }
        
        return mappedBuffer;
    }

    void Device::unmapBuffer(IBuffer* _b)
    {
        Buffer* b = checked_cast<Buffer*>(_b);

        b->resource->Unmap(0, nullptr);
    }

    MemoryRequirements Device::getBufferMemoryRequirements(IBuffer* _buffer)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);

        D3D12_RESOURCE_ALLOCATION_INFO allocInfo = m_Context.device->GetResourceAllocationInfo(1, 1, &buffer->resourceDesc);

        MemoryRequirements memReq;
        memReq.alignment = allocInfo.Alignment;
        memReq.size = allocInfo.SizeInBytes;
        return memReq;
    }

    bool Device::bindBufferMemory(IBuffer* _buffer, IHeap* _heap, uint64_t offset)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);
        Heap* heap = checked_cast<Heap*>(_heap);

        if (buffer->resource)
            return false; // already bound

        if (!buffer->desc.isVirtual)
            return false; // not supported

        D3D12_RESOURCE_STATES initialState = convertResourceStates(buffer->desc.initialState);
        if (initialState != D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)
            initialState = D3D12_RESOURCE_STATE_COMMON;

        HRESULT hr = m_Context.device->CreatePlacedResource(
            heap->heap, offset,
            &buffer->resourceDesc,
            initialState,
            nullptr,
            IID_PPV_ARGS(&buffer->resource));

        if (FAILED(hr))
        {
            std::stringstream ss;
            ss << "Failed to create placed buffer " << utils::DebugNameToString(buffer->desc.debugName) << ", error code = 0x";
            ss.setf(std::ios::hex, std::ios::basefield);
            ss << hr;
            m_Context.error(ss.str());

            return false;
        }

        buffer->heap = heap;
        buffer->postCreate();

        return true;
    }

    nvrhi::BufferHandle Device::createHandleForNativeBuffer(ObjectType objectType, Object _buffer, const BufferDesc& desc)
    {
        if (_buffer.pointer == nullptr)
            return nullptr;

        if (objectType != ObjectTypes::D3D12_Resource)
            return nullptr;

        ID3D12Resource* pResource = static_cast<ID3D12Resource*>(_buffer.pointer);

        Buffer* buffer = new Buffer(m_Context, m_Resources, desc);
        buffer->resource = pResource;
        
        buffer->postCreate();

        return BufferHandle::Create(buffer);
    }

    void Buffer::createCBV(size_t descriptor, BufferRange range) const
    {
        assert(desc.isConstantBuffer);

        range = range.resolve(desc);
        assert(range.byteSize <= UINT_MAX);

        D3D12_CONSTANT_BUFFER_VIEW_DESC viewDesc;
        viewDesc.BufferLocation = resource->GetGPUVirtualAddress() + range.byteOffset;
        viewDesc.SizeInBytes = align((UINT)range.byteSize, c_ConstantBufferOffsetSizeAlignment);
        m_Context.device->CreateConstantBufferView(&viewDesc, { descriptor });
    }
    
    void Buffer::createNullSRV(size_t descriptor, Format format, const Context& context)
    {
        const DxgiFormatMapping& mapping = getDxgiFormatMapping(format == Format::UNKNOWN ? Format::R32_UINT : format);

        D3D12_SHADER_RESOURCE_VIEW_DESC viewDesc = {};
        viewDesc.Format = mapping.srvFormat;
        viewDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        viewDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        context.device->CreateShaderResourceView(nullptr, &viewDesc, { descriptor });
    }

    void Buffer::createSRV(size_t descriptor, Format format, BufferRange range, ResourceType type) const
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC viewDesc = {};

        viewDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        viewDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        
        if (format == Format::UNKNOWN)
        {
            format = desc.format;
        }

        range = range.resolve(desc);

        switch (type)  // NOLINT(clang-diagnostic-switch-enum)
        {
        case ResourceType::StructuredBuffer_SRV:
            assert(desc.structStride != 0);
            viewDesc.Format = DXGI_FORMAT_UNKNOWN;
            viewDesc.Buffer.FirstElement = range.byteOffset / desc.structStride;
            viewDesc.Buffer.NumElements = (UINT)(range.byteSize / desc.structStride);
            viewDesc.Buffer.StructureByteStride = desc.structStride;
            break;

        case ResourceType::RawBuffer_SRV:
            viewDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            viewDesc.Buffer.FirstElement = range.byteOffset / 4;
            viewDesc.Buffer.NumElements = (UINT)(range.byteSize / 4);
            viewDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            break;

        case ResourceType::TypedBuffer_SRV:
        {
            assert(format != Format::UNKNOWN);
            const DxgiFormatMapping& mapping = getDxgiFormatMapping(format);
            const FormatInfo& formatInfo = getFormatInfo(format);

            viewDesc.Format = mapping.srvFormat;
            viewDesc.Buffer.FirstElement = range.byteOffset / formatInfo.bytesPerBlock;
            viewDesc.Buffer.NumElements = (UINT)(range.byteSize / formatInfo.bytesPerBlock);
            break;
        }

        default:
            utils::InvalidEnum();
            return;
        }

        m_Context.device->CreateShaderResourceView(resource, &viewDesc, { descriptor });
    }

    void Buffer::createNullUAV(size_t descriptor, Format format, const Context& context)
    {
        const DxgiFormatMapping& mapping = getDxgiFormatMapping(format == Format::UNKNOWN ? Format::R32_UINT : format);

        D3D12_UNORDERED_ACCESS_VIEW_DESC viewDesc = {};
        viewDesc.Format = mapping.srvFormat;
        viewDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        context.device->CreateUnorderedAccessView(nullptr, nullptr, &viewDesc, { descriptor });
    }

    void Buffer::createUAV(size_t descriptor, Format format, BufferRange range, ResourceType type) const
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC viewDesc = {};

        viewDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        
        if (format == Format::UNKNOWN)
        {
            format = desc.format;
        }

        range = range.resolve(desc);

        switch (type)  // NOLINT(clang-diagnostic-switch-enum)
        {
        case ResourceType::StructuredBuffer_UAV:
            assert(desc.structStride != 0);
            viewDesc.Format = DXGI_FORMAT_UNKNOWN;
            viewDesc.Buffer.FirstElement = range.byteOffset / desc.structStride;
            viewDesc.Buffer.NumElements = (UINT)(range.byteSize / desc.structStride);
            viewDesc.Buffer.StructureByteStride = desc.structStride;
            break;

        case ResourceType::RawBuffer_UAV:
            viewDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            viewDesc.Buffer.FirstElement = range.byteOffset / 4;
            viewDesc.Buffer.NumElements = (UINT)(range.byteSize / 4);
            viewDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            break;

        case ResourceType::TypedBuffer_UAV:
        {
            assert(format != Format::UNKNOWN);
            const DxgiFormatMapping& mapping = getDxgiFormatMapping(format);
            const FormatInfo& formatInfo = getFormatInfo(format);

            viewDesc.Format = mapping.srvFormat;
            viewDesc.Buffer.FirstElement = range.byteOffset / formatInfo.bytesPerBlock;
            viewDesc.Buffer.NumElements = (UINT)(range.byteSize / formatInfo.bytesPerBlock);
            break;
        }

        default: 
            utils::InvalidEnum();
            return;
        }

        m_Context.device->CreateUnorderedAccessView(resource, nullptr, &viewDesc, { descriptor });
    }
    
    void CommandList::writeBuffer(IBuffer* _b, const void * data, size_t dataSize, uint64_t destOffsetBytes)
    {
        Buffer* buffer = checked_cast<Buffer*>(_b);

        void* cpuVA;
        D3D12_GPU_VIRTUAL_ADDRESS gpuVA;
        ID3D12Resource* uploadBuffer;
        size_t offsetInUploadBuffer;
        if (!m_UploadManager.suballocateBuffer(dataSize, nullptr, &uploadBuffer, &offsetInUploadBuffer, &cpuVA, &gpuVA, 
            m_RecordingVersion, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT))
        {
            m_Context.error("Couldn't suballocate an upload buffer");
            return;
        }

        if (uploadBuffer != m_CurrentUploadBuffer)
        {
            m_Instance->referencedNativeResources.push_back(uploadBuffer);
            m_CurrentUploadBuffer = uploadBuffer;
        }

        memcpy(cpuVA, data, dataSize);

        if (buffer->desc.isVolatile)
        {
            m_VolatileConstantBufferAddresses[buffer] = gpuVA;
            m_AnyVolatileBufferWrites = true;
        }
        else
        {
            if (m_EnableAutomaticBarriers)
            {
                requireBufferState(buffer, ResourceStates::CopyDest);
            }
            commitBarriers();

            m_Instance->referencedResources.push_back(buffer);

            m_ActiveCommandList->commandList->CopyBufferRegion(buffer->resource, destOffsetBytes, uploadBuffer, offsetInUploadBuffer, dataSize);
        }
    }

    void CommandList::clearBufferUInt(IBuffer* _b, uint32_t clearValue)
    {
        Buffer* b = checked_cast<Buffer*>(_b);

        if (!b->desc.canHaveUAVs)
        {
            std::stringstream ss;
            ss << "Cannot clear buffer " << utils::DebugNameToString(b->desc.debugName)
               << " because it was created with canHaveUAVs = false";
            m_Context.error(ss.str());
            return;
        }

        if (m_EnableAutomaticBarriers)
        {
            requireBufferState(b, ResourceStates::UnorderedAccess);
        }
        commitBarriers();

        commitDescriptorHeaps();

        DescriptorIndex clearUAV = b->getClearUAV();
        assert(clearUAV != c_InvalidDescriptorIndex);

        m_Instance->referencedResources.push_back(b);

        const uint32_t values[4] = { clearValue, clearValue, clearValue, clearValue };
        m_ActiveCommandList->commandList->ClearUnorderedAccessViewUint(
            m_Resources.shaderResourceViewHeap.getGpuHandle(clearUAV),
            m_Resources.shaderResourceViewHeap.getCpuHandle(clearUAV),
            b->resource, values, 0, nullptr);
    }

    void CommandList::copyBuffer(IBuffer* _dest, uint64_t destOffsetBytes, IBuffer* _src, uint64_t srcOffsetBytes, uint64_t dataSizeBytes)
    {
        Buffer* dest = checked_cast<Buffer*>(_dest);
        Buffer* src = checked_cast<Buffer*>(_src);

        if (m_EnableAutomaticBarriers)
        {
            requireBufferState(dest, ResourceStates::CopyDest);
            requireBufferState(src, ResourceStates::CopySource);
        }
        commitBarriers();

        if(src->desc.cpuAccess != CpuAccessMode::None)
            m_Instance->referencedStagingBuffers.push_back(src);
        else
            m_Instance->referencedResources.push_back(src);

        if (dest->desc.cpuAccess != CpuAccessMode::None)
            m_Instance->referencedStagingBuffers.push_back(dest);
        else
            m_Instance->referencedResources.push_back(dest);

        m_ActiveCommandList->commandList->CopyBufferRegion(dest->resource, destOffsetBytes, src->resource, srcOffsetBytes, dataSizeBytes);
    }

} // namespace nvrhi::d3d12
