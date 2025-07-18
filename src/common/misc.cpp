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

#include <nvrhi/nvrhi.h>
#include <nvrhi/utils.h>
#include <cmath>

namespace nvrhi
{
    // Do not move this function into the header.
    bool verifyHeaderVersion(uint32_t version)
    {
        return version == c_HeaderVersion;
    }

    TextureSlice TextureSlice::resolve(const TextureDesc& desc) const
    {
        TextureSlice ret(*this);

        assert(mipLevel < desc.mipLevels);

        if (width == uint32_t(-1))
            ret.width = std::max(desc.width >> mipLevel, 1u);

        if (height == uint32_t(-1))
            ret.height = std::max(desc.height >> mipLevel, 1u);

        if (depth == uint32_t(-1))
        {
            if (desc.dimension == TextureDimension::Texture3D)
                ret.depth = std::max(desc.depth >> mipLevel, 1u);
            else
                ret.depth = 1;
        }

        // [rlaw] BEGIN
        const FormatInfo& formatInfo = getFormatInfo(desc.format);
        const uint32_t blockSize = formatInfo.blockSize;
        const bool bIsCompressedFormat = formatInfo.blockSize != 1;

        // ensure that the width and height are at least the block size for compressed formats
        ret.width = std::max(ret.width, blockSize);
        ret.height = std::max(ret.height, blockSize);

        // If the texture is compressed, we need to round the width and height up to the nearest block size
        if (bIsCompressedFormat)
        {
            ret.width = (ret.width + blockSize - 1) / blockSize * blockSize;
            ret.height = (ret.height + blockSize - 1) / blockSize * blockSize;
        }
        // [rlaw] END

        return ret;
    }
        
    TextureSubresourceSet TextureSubresourceSet::resolve(const TextureDesc& desc, bool singleMipLevel) const
    {
        TextureSubresourceSet ret;
        ret.baseMipLevel = baseMipLevel;

        if (singleMipLevel)
        {
            ret.numMipLevels = 1;
        }
        else
        {
            int lastMipLevelPlusOne = std::min(baseMipLevel + numMipLevels, desc.mipLevels);
            ret.numMipLevels = MipLevel(std::max(0u, lastMipLevelPlusOne - baseMipLevel));
        }

        switch (desc.dimension)  // NOLINT(clang-diagnostic-switch-enum)
        {
        case TextureDimension::Texture1DArray:
        case TextureDimension::Texture2DArray:
        case TextureDimension::TextureCube:
        case TextureDimension::TextureCubeArray:
        case TextureDimension::Texture2DMSArray: {
            ret.baseArraySlice = baseArraySlice;
            int lastArraySlicePlusOne = std::min(baseArraySlice + numArraySlices, desc.arraySize);
            ret.numArraySlices = ArraySlice(std::max(0u, lastArraySlicePlusOne - baseArraySlice));
            break;
        }
        default: 
            ret.baseArraySlice = 0;
            ret.numArraySlices = 1;
            break;
        }

        return ret;
    }

    bool TextureSubresourceSet::isEntireTexture(const TextureDesc& desc) const
    {
        if (baseMipLevel > 0u || baseMipLevel + numMipLevels < desc.mipLevels)
            return false;

        switch (desc.dimension)  // NOLINT(clang-diagnostic-switch-enum)
        {
        case TextureDimension::Texture1DArray:
        case TextureDimension::Texture2DArray:
        case TextureDimension::TextureCube:
        case TextureDimension::TextureCubeArray:
        case TextureDimension::Texture2DMSArray: 
            if (baseArraySlice > 0u || baseArraySlice + numArraySlices < desc.arraySize)
                return false;
        default:
            return true;
        }
    }
    
    BufferRange BufferRange::resolve(const BufferDesc& desc) const
    {
        BufferRange result;
        result.byteOffset = std::min(byteOffset, desc.byteSize);
        if (byteSize == 0)
            result.byteSize = desc.byteSize - result.byteOffset;
        else
            result.byteSize = std::min(byteSize, desc.byteSize - result.byteOffset);
        return result;
    }
    
    bool BlendState::RenderTarget::usesConstantColor() const
    {
        return srcBlend == BlendFactor::ConstantColor || srcBlend == BlendFactor::OneMinusConstantColor ||
            destBlend == BlendFactor::ConstantColor || destBlend == BlendFactor::OneMinusConstantColor ||
            srcBlendAlpha == BlendFactor::ConstantColor || srcBlendAlpha == BlendFactor::OneMinusConstantColor ||
            destBlendAlpha == BlendFactor::ConstantColor || destBlendAlpha == BlendFactor::OneMinusConstantColor;
    }

    bool BlendState::usesConstantColor(uint32_t numTargets) const
    {
        for (uint32_t rt = 0; rt < numTargets; rt++)
        {
            if (targets[rt].usesConstantColor())
                return true;
        }

        return false;
    }
    
    FramebufferInfo::FramebufferInfo(const FramebufferDesc& desc)
    {
        for (size_t i = 0; i < desc.colorAttachments.size(); i++)
        {
            const FramebufferAttachment& attachment = desc.colorAttachments[i];
            colorFormats.push_back(attachment.format == Format::UNKNOWN && attachment.texture ? attachment.texture->getDesc().format : attachment.format);
        }

        if (desc.depthAttachment.valid())
        {
            const TextureDesc& textureDesc = desc.depthAttachment.texture->getDesc();
            depthFormat = textureDesc.format;
            sampleCount = textureDesc.sampleCount;
            sampleQuality = textureDesc.sampleQuality;
        }
        else if (!desc.colorAttachments.empty() && desc.colorAttachments[0].valid())
        {
            const TextureDesc& textureDesc = desc.colorAttachments[0].texture->getDesc();
            sampleCount = textureDesc.sampleCount;
            sampleQuality = textureDesc.sampleQuality;
        }
    }

    FramebufferInfoEx::FramebufferInfoEx(const FramebufferDesc& desc)
        : FramebufferInfo(desc)
    {
        if (desc.depthAttachment.valid())
        {
            const TextureDesc& textureDesc = desc.depthAttachment.texture->getDesc();
            width = std::max(textureDesc.width >> desc.depthAttachment.subresources.baseMipLevel, 1u);
            height = std::max(textureDesc.height >> desc.depthAttachment.subresources.baseMipLevel, 1u);
        }
        else if (!desc.colorAttachments.empty() && desc.colorAttachments[0].valid())
        {
            const TextureDesc& textureDesc = desc.colorAttachments[0].texture->getDesc();
            width = std::max(textureDesc.width >> desc.colorAttachments[0].subresources.baseMipLevel, 1u);
            height = std::max(textureDesc.height >> desc.colorAttachments[0].subresources.baseMipLevel, 1u);
        }
    }

    void ICommandList::setResourceStatesForFramebuffer(IFramebuffer* framebuffer)
    {
        const FramebufferDesc& desc = framebuffer->getDesc();

        for (const auto& attachment : desc.colorAttachments)
        {
            setTextureState(attachment.texture, attachment.subresources,
                ResourceStates::RenderTarget);
        }

        if (desc.depthAttachment.valid())
        {
            setTextureState(desc.depthAttachment.texture, desc.depthAttachment.subresources,
                desc.depthAttachment.isReadOnly ? ResourceStates::DepthRead : ResourceStates::DepthWrite);
        }
    }
    
    size_t coopvec::getDataTypeSize(coopvec::DataType type)
    {
        switch (type)
        {
        case coopvec::DataType::UInt8:
        case coopvec::DataType::SInt8:
            return 1;
        case coopvec::DataType::UInt8Packed:
        case coopvec::DataType::SInt8Packed:
            // Not sure if this is correct or even relevant because packed types
            // cannot be used in matrices accessible from the host side.
            return 1;
        case coopvec::DataType::UInt16:
        case coopvec::DataType::SInt16:
            return 2;
        case coopvec::DataType::UInt32:
        case coopvec::DataType::SInt32:
            return 4;
        case coopvec::DataType::UInt64:
        case coopvec::DataType::SInt64:
            return 8;
        case coopvec::DataType::FloatE4M3:
        case coopvec::DataType::FloatE5M2:
            return 1;
        case coopvec::DataType::Float16:
        case coopvec::DataType::BFloat16:
            return 2;
        case coopvec::DataType::Float32:
            return 4;
        case coopvec::DataType::Float64:
            return 8;
        default:
            utils::InvalidEnum();
            return 0;
        }
    }

    size_t coopvec::getOptimalMatrixStride(coopvec::DataType type, coopvec::MatrixLayout layout, uint32_t rows, uint32_t columns)
    {
        size_t const dataTypeSize = coopvec::getDataTypeSize(type);
        
        switch (layout)
        {
        case coopvec::MatrixLayout::RowMajor:
            return dataTypeSize * columns;
            break;
        case coopvec::MatrixLayout::ColumnMajor:
            return dataTypeSize * rows;
            break;
        default:
            return 0;
        }
    }

} // namespace nvrhi