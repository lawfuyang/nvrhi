#include "test-harness.h"

#include <nvrhi/common/misc.h>
#include <nvrhi/common/resourcebindingmap.h>
#include <nvrhi/utils.h>

#include <array>
#include <string>
#include <unordered_map>

namespace
{
    template <typename TInterface>
    struct FakeResource : TInterface
    {
        unsigned long references = 1;

        unsigned long AddRef() override { return ++references; }
        unsigned long Release() override { return --references; }
        unsigned long GetRefCount() override { return references; }
    };

    struct FakeTexture : FakeResource<nvrhi::ITexture>
    {
        nvrhi::TextureDesc desc;

        const nvrhi::TextureDesc& getDesc() const override { return desc; }

        nvrhi::Object getNativeView(nvrhi::ObjectType, nvrhi::Format,
            nvrhi::TextureSubresourceSet, nvrhi::TextureDimension, bool,
            std::optional<nvrhi::ComponentMapping>) override
        {
            return nullptr;
        }
    };

    struct FakeBuffer : FakeResource<nvrhi::IBuffer>
    {
        nvrhi::BufferDesc desc;

        const nvrhi::BufferDesc& getDesc() const override { return desc; }
        nvrhi::GpuVirtualAddress getGpuVirtualAddress() const override { return 0; }
    };

    void checkContains(const std::string& text, const char* needle)
    {
        if (text.find(needle) == std::string::npos)
        {
            nvrhi_test::reportFailure(__FILE__, __LINE__,
                std::string("expected \"") + needle + "\" in \"" + text + "\"");
        }
    }

} // namespace

NVRHI_TEST_CASE("align rounds up to the requested alignment")
{
    CHECK_EQ(nvrhi::align(0u, 256u), 0u);
    CHECK_EQ(nvrhi::align(1u, 256u), 256u);
    CHECK_EQ(nvrhi::align(256u, 256u), 256u);
    CHECK_EQ(nvrhi::align(257u, 256u), 512u);
    CHECK_EQ(nvrhi::align(300u, 64u), 320u);
    CHECK_EQ(nvrhi::align(5u, 1u), 5u);
    CHECK_EQ(nvrhi::align<uint64_t>(4097, 4096), 8192ull);}

NVRHI_TEST_CASE("arraysAreDifferent compares sizes and elements")
{
    const std::array<int, 3> a = { 1, 2, 3 };
    const std::array<int, 3> equal = { 1, 2, 3 };
    const std::array<int, 3> different = { 1, 2, 4 };

    CHECK(!nvrhi::arraysAreDifferent(a, equal));
    CHECK(nvrhi::arraysAreDifferent(a, different));

    const nvrhi::static_vector<int, 4> shorter = { 1, 2 };
    const nvrhi::static_vector<int, 4> longer = { 1, 2, 3 };
    CHECK(nvrhi::arraysAreDifferent(shorter, longer));
}

NVRHI_TEST_CASE("arrayDifferenceMask reports differing elements")
{
    const nvrhi::static_vector<int, 4> a = { 1, 2, 3 };
    const nvrhi::static_vector<int, 4> equal = { 1, 2, 3 };
    const nvrhi::static_vector<int, 4> different = { 1, 9, 3 };
    const nvrhi::static_vector<int, 4> shorter = { 1, 2 };

    CHECK_EQ(nvrhi::arrayDifferenceMask(a, equal), 0u);
    CHECK_EQ(nvrhi::arrayDifferenceMask(a, different), 2u);
    CHECK_EQ(nvrhi::arrayDifferenceMask(a, shorter), ~0u);
}

NVRHI_TEST_CASE("hash_to_u32 folds both halves of a size_t")
{
    CHECK_EQ(nvrhi::hash_to_u32(0), 0u);
    CHECK_EQ(nvrhi::hash_to_u32(0x12345678u), 0x12345678u);

    const size_t folded = size_t(0xAAAAAAAA11111111ull);
    CHECK_EQ(nvrhi::hash_to_u32(folded), 0xBBBBBBBBu);
}

NVRHI_TEST_CASE("static_vector stores elements up to its capacity")
{
    nvrhi::static_vector<int, 4> values;
    CHECK(values.empty());
    CHECK_EQ(values.size(), size_t(0));
    CHECK_EQ(values.max_size(), size_t(4));

    values.push_back(1);
    values.push_back(2);
    values.push_back(3);
    CHECK_EQ(values.size(), size_t(3));
    CHECK_EQ(values[0], 1);
    CHECK_EQ(values[1], 2);
    CHECK_EQ(values.back(), 3);

    values.pop_back();
    CHECK_EQ(values.size(), size_t(2));

    values.resize(4);
    CHECK_EQ(values.size(), size_t(4));
    CHECK_EQ(values[2], 0);
    CHECK_EQ(values[3], 0);

    values.resize(1);
    CHECK_EQ(values.size(), size_t(1));

    int& emplaced = values.emplace_back();
    emplaced = 42;
    CHECK_EQ(values.size(), size_t(2));
    CHECK_EQ(values.back(), 42);

    values.fill(7);
    CHECK_EQ(values.size(), size_t(4));
    for (size_t index = 0; index < values.size(); ++index)
        CHECK_EQ(values[index], 7);

    nvrhi::static_vector<int, 4> swapped = { 9, 8 };
    values.swap(swapped);
    CHECK_EQ(values.size(), size_t(2));
    CHECK_EQ(values[0], 9);
    CHECK_EQ(swapped.size(), size_t(4));

    const nvrhi::static_vector<int, 4> initialized = { 4, 5, 6 };
    CHECK_EQ(initialized.size(), size_t(3));
    CHECK_EQ(initialized[2], 6);
}

NVRHI_TEST_CASE("Color and geometry helpers keep their declared defaults")
{
    const nvrhi::Color transparent;
    CHECK_EQ(transparent.r, 0.f);
    CHECK_EQ(transparent.a, 0.f);

    const nvrhi::Color white(1.f);
    CHECK(white == nvrhi::Color(1.f, 1.f, 1.f, 1.f));
    CHECK(white != nvrhi::Color(1.f, 1.f, 1.f, 0.f));

    const nvrhi::Viewport defaultViewport;
    CHECK_EQ(defaultViewport.minZ, 0.f);
    CHECK_EQ(defaultViewport.maxZ, 1.f);

    const nvrhi::Viewport sizedViewport(640.f, 480.f);
    CHECK_EQ(sizedViewport.width(), 640.f);
    CHECK_EQ(sizedViewport.height(), 480.f);
    CHECK(sizedViewport == nvrhi::Viewport(0.f, 640.f, 0.f, 480.f, 0.f, 1.f));

    const nvrhi::Rect rect(defaultViewport);
    CHECK_EQ(rect.minX, 0);
    CHECK_EQ(rect.maxX, 0);

    const nvrhi::Rect rounded(nvrhi::Viewport(0.2f, 10.7f, -1.2f, 3.9f, 0.f, 1.f));
    CHECK_EQ(rounded.minX, 0);
    CHECK_EQ(rounded.maxX, 11);
    CHECK_EQ(rounded.minY, -2);
    CHECK_EQ(rounded.maxY, 4);
    CHECK_EQ(rounded.width(), 11);
    CHECK_EQ(rounded.height(), 6);
}

NVRHI_TEST_CASE("component mappings pack and resolve")
{
    const nvrhi::ComponentMapping identity;
    CHECK(identity.isIdentity());
    CHECK_EQ(identity.pack(), uint16_t((1u << 3) | (2u << 6) | (3u << 9)));
    const nvrhi::ComponentMapping allZero{
        nvrhi::ComponentSwizzle::Zero, nvrhi::ComponentSwizzle::Zero,
        nvrhi::ComponentSwizzle::Zero, nvrhi::ComponentSwizzle::Zero };
    CHECK_EQ(allZero.pack(), uint16_t(4u | (4u << 3) | (4u << 6) | (4u << 9)));

    for (uint32_t r = 0; r < 6; ++r)
    {
        for (uint32_t g = 0; g < 6; ++g)
        {
            for (uint32_t b = 0; b < 6; ++b)
            {
                for (uint32_t a = 0; a < 6; ++a)
                {
                    const nvrhi::ComponentMapping mapping{
                        nvrhi::ComponentSwizzle(r), nvrhi::ComponentSwizzle(g),
                        nvrhi::ComponentSwizzle(b), nvrhi::ComponentSwizzle(a) };
                    CHECK(nvrhi::ComponentMapping::unpack(mapping.pack()) == mapping);
                }
            }
        }
    }

    const nvrhi::ComponentMapping bgra{
        nvrhi::ComponentSwizzle::B, nvrhi::ComponentSwizzle::G,
        nvrhi::ComponentSwizzle::R, nvrhi::ComponentSwizzle::A };
    CHECK(!bgra.isIdentity());

    CHECK_EQ(nvrhi::packComponentMapping(std::nullopt), 0u);
    const uint16_t packedIdentity = nvrhi::packComponentMapping(nvrhi::ComponentMapping());
    CHECK((packedIdentity & nvrhi::c_ComponentMappingExplicit) != 0);
    CHECK_EQ(nvrhi::resolveComponentMapping(uint16_t(0), bgra), bgra);
    CHECK(nvrhi::resolveComponentMapping(packedIdentity, bgra).isIdentity());
    CHECK(nvrhi::resolveComponentMapping(bgra, identity) == bgra);
}

NVRHI_TEST_CASE("format information is consistent for every format")
{
    uint32_t depthFormats = 0;
    uint32_t stencilFormats = 0;
    uint32_t srgbFormats = 0;
    uint32_t compressedFormats = 0;

    for (uint32_t index = 0; index < uint32_t(nvrhi::Format::COUNT); ++index)
    {
        const nvrhi::Format format = nvrhi::Format(index);
        const nvrhi::FormatInfo& info = nvrhi::getFormatInfo(format);

        CHECK(info.format == format);
        REQUIRE(info.name != nullptr);
        if (format != nvrhi::Format::UNKNOWN)
        {
            CHECK(info.blockSize >= 1);
            CHECK(info.bytesPerBlock > 0);
        }
        if (info.blockSize > 1)
            ++compressedFormats;
        if (info.hasDepth)
            ++depthFormats;
        if (info.hasStencil)
            ++stencilFormats;
        if (info.isSRGB)
            ++srgbFormats;
    }

    CHECK_EQ(depthFormats, 4u);
    CHECK_EQ(stencilFormats, 4u);
    CHECK_EQ(srgbFormats, 7u);
    CHECK_EQ(compressedFormats, 14u);

    const nvrhi::FormatInfo& unknown = nvrhi::getFormatInfo(nvrhi::Format::UNKNOWN);
    CHECK_EQ(unknown.bytesPerBlock, uint8_t(0));
    CHECK_EQ(unknown.name, std::string("UNKNOWN"));

    const nvrhi::FormatInfo& rgba8 = nvrhi::getFormatInfo(nvrhi::Format::RGBA8_UNORM);
    CHECK_EQ(rgba8.bytesPerBlock, uint8_t(4));
    CHECK_EQ(rgba8.blockSize, uint8_t(1));
    CHECK(rgba8.kind == nvrhi::FormatKind::Normalized);
    CHECK(rgba8.hasRed && rgba8.hasGreen && rgba8.hasBlue && rgba8.hasAlpha);
    CHECK(!rgba8.hasDepth && !rgba8.hasStencil);

    const nvrhi::FormatInfo& bc3 = nvrhi::getFormatInfo(nvrhi::Format::BC3_UNORM_SRGB);
    CHECK_EQ(bc3.bytesPerBlock, uint8_t(16));
    CHECK_EQ(bc3.blockSize, uint8_t(4));
    CHECK(bc3.isSRGB);

    const nvrhi::FormatInfo& depthStencil = nvrhi::getFormatInfo(nvrhi::Format::D32S8);
    CHECK(depthStencil.hasDepth && depthStencil.hasStencil);
    CHECK(depthStencil.kind == nvrhi::FormatKind::DepthStencil);

    const nvrhi::FormatInfo& outOfRange = nvrhi::getFormatInfo(nvrhi::Format(200));
    CHECK(outOfRange.format == nvrhi::Format::UNKNOWN);
    CHECK_EQ(outOfRange.name, std::string("UNKNOWN"));

    CHECK_EQ(nvrhi::utils::FormatToString(nvrhi::Format::R8_UNORM), std::string("R8_UNORM"));
}

NVRHI_TEST_CASE("TextureSlice resolves whole mip levels")
{
    nvrhi::TextureDesc desc;
    desc.setWidth(256).setHeight(128).setDepth(1).setMipLevels(4)
        .setFormat(nvrhi::Format::RGBA8_UNORM);

    const nvrhi::TextureSlice whole = nvrhi::TextureSlice().resolve(desc);
    CHECK_EQ(whole.width, 256u);
    CHECK_EQ(whole.height, 128u);
    CHECK_EQ(whole.depth, 1u);

    const nvrhi::TextureSlice mip2 = nvrhi::TextureSlice().setMipLevel(2).resolve(desc);
    CHECK_EQ(mip2.width, 64u);
    CHECK_EQ(mip2.height, 32u);

    const nvrhi::TextureSlice explicitSize =
        nvrhi::TextureSlice().setOrigin(3, 4, 0).setSize(5, 6, 7).resolve(desc);
    CHECK_EQ(explicitSize.x, 3u);
    CHECK_EQ(explicitSize.y, 4u);
    CHECK_EQ(explicitSize.width, 5u);
    CHECK_EQ(explicitSize.height, 6u);
    CHECK_EQ(explicitSize.depth, 7u);

    nvrhi::TextureDesc volume;
    volume.setWidth(64).setHeight(32).setDepth(16).setMipLevels(5)
        .setFormat(nvrhi::Format::RGBA8_UNORM)
        .setDimension(nvrhi::TextureDimension::Texture3D);

    const nvrhi::TextureSlice volumeMip2 = nvrhi::TextureSlice().setMipLevel(2).resolve(volume);
    CHECK_EQ(volumeMip2.width, 16u);
    CHECK_EQ(volumeMip2.height, 8u);
    CHECK_EQ(volumeMip2.depth, 4u);

    nvrhi::TextureDesc packed;
    packed.setWidth(6).setHeight(8).setMipLevels(6)
        .setFormat(nvrhi::Format::BC1_UNORM);

    const nvrhi::TextureSlice packedMip0 = nvrhi::TextureSlice().resolve(packed);
    CHECK_EQ(packedMip0.width, 8u);
    CHECK_EQ(packedMip0.height, 8u);

    const nvrhi::TextureSlice packedMip2 = nvrhi::TextureSlice().setMipLevel(2).resolve(packed);
    CHECK_EQ(packedMip2.width, 4u);
    CHECK_EQ(packedMip2.height, 4u);

    nvrhi::TextureDesc zeroWidth;
    zeroWidth.setWidth(0).setHeight(0).setFormat(nvrhi::Format::RGBA8_UNORM);
    const nvrhi::TextureSlice clamped = nvrhi::TextureSlice().resolve(zeroWidth);
    CHECK_EQ(clamped.width, 1u);
    CHECK_EQ(clamped.height, 1u);
}

NVRHI_TEST_CASE("TextureSubresourceSet resolves and detects entire textures")
{
    nvrhi::TextureDesc arrayDesc;
    arrayDesc.setWidth(64).setHeight(64).setMipLevels(4).setArraySize(3)
        .setFormat(nvrhi::Format::RGBA8_UNORM)
        .setDimension(nvrhi::TextureDimension::Texture2DArray);

    const nvrhi::TextureSubresourceSet all = nvrhi::AllSubresources.resolve(arrayDesc, false);
    CHECK_EQ(all.baseMipLevel, 0u);
    CHECK_EQ(all.numMipLevels, 4u);
    CHECK_EQ(all.baseArraySlice, 0u);
    CHECK_EQ(all.numArraySlices, 3u);
    CHECK(nvrhi::AllSubresources.isEntireTexture(arrayDesc));

    const nvrhi::TextureSubresourceSet subset =
        nvrhi::TextureSubresourceSet(1, 2, 2, 5).resolve(arrayDesc, false);
    CHECK_EQ(subset.baseMipLevel, 1u);
    CHECK_EQ(subset.numMipLevels, 2u);
    CHECK_EQ(subset.baseArraySlice, 2u);
    CHECK_EQ(subset.numArraySlices, 1u);
    CHECK(!subset.isEntireTexture(arrayDesc));

    const nvrhi::TextureSubresourceSet single =
        nvrhi::AllSubresources.resolve(arrayDesc, true);
    CHECK_EQ(single.numMipLevels, 1u);

    const nvrhi::TextureSubresourceSet outOfRange =
        nvrhi::TextureSubresourceSet(4, 2, 0, 1).resolve(arrayDesc, false);
    CHECK_EQ(outOfRange.numMipLevels, 0u);

    nvrhi::TextureDesc flatDesc = arrayDesc;
    flatDesc.setDimension(nvrhi::TextureDimension::Texture2D);
    const nvrhi::TextureSubresourceSet flat = nvrhi::AllSubresources.resolve(flatDesc, false);
    CHECK_EQ(flat.baseArraySlice, 0u);
    CHECK_EQ(flat.numArraySlices, 1u);

    CHECK_EQ(nvrhi::TextureSubresourceSet::AllMipLevels, uint32_t(-1));
    CHECK_EQ(nvrhi::TextureSubresourceSet::AllArraySlices, uint32_t(-1));
    CHECK(nvrhi::TextureSubresourceSet(1, 2, 3, 4) == nvrhi::TextureSubresourceSet(1, 2, 3, 4));
    CHECK(nvrhi::TextureSubresourceSet(1, 2, 3, 4) != nvrhi::TextureSubresourceSet(1, 2, 3, 5));
}

NVRHI_TEST_CASE("BufferRange clamps offsets and sizes")
{
    nvrhi::BufferDesc desc;
    desc.byteSize = 4096;

    const nvrhi::BufferRange entire = nvrhi::EntireBuffer.resolve(desc);
    CHECK_EQ(entire.byteOffset, 0ull);
    CHECK_EQ(entire.byteSize, 4096ull);
    CHECK(nvrhi::EntireBuffer.isEntireBuffer(desc));
    CHECK(nvrhi::BufferRange(0, 4096).isEntireBuffer(desc));
    CHECK(!nvrhi::BufferRange(0, 4095).isEntireBuffer(desc));
    CHECK(!nvrhi::BufferRange(1, 4095).isEntireBuffer(desc));

    const nvrhi::BufferRange tail = nvrhi::BufferRange(100, 0).resolve(desc);
    CHECK_EQ(tail.byteOffset, 100ull);
    CHECK_EQ(tail.byteSize, 3996ull);

    const nvrhi::BufferRange clamped = nvrhi::BufferRange(4000, 500).resolve(desc);
    CHECK_EQ(clamped.byteOffset, 4000ull);
    CHECK_EQ(clamped.byteSize, 96ull);

    const nvrhi::BufferRange past = nvrhi::BufferRange(5000, 10).resolve(desc);
    CHECK_EQ(past.byteOffset, 4096ull);
    CHECK_EQ(past.byteSize, 0ull);
}

NVRHI_TEST_CASE("blend states compare and detect constant colors")
{
    nvrhi::BlendState::RenderTarget target;
    CHECK(!target.usesConstantColor());
    CHECK(target.colorWriteMask == nvrhi::ColorMask::All);

    target.setSrcBlend(nvrhi::BlendFactor::ConstantColor);
    CHECK(target.usesConstantColor());

    target.setSrcBlend(nvrhi::BlendFactor::One);
    target.setDestBlendAlpha(nvrhi::BlendFactor::OneMinusConstantColor);
    CHECK(target.usesConstantColor());

    target.setDestBlendAlpha(nvrhi::BlendFactor::InvSrcAlpha);
    CHECK(!target.usesConstantColor());

    nvrhi::BlendState blend;
    CHECK(!blend.usesConstantColor(1));
    blend.setRenderTarget(1, nvrhi::BlendState::RenderTarget()
        .setSrcBlend(nvrhi::BlendFactor::ConstantColor));
    CHECK(!blend.usesConstantColor(1));
    CHECK(blend.usesConstantColor(2));

    nvrhi::BlendState other = blend;
    CHECK(other == blend);
    other.enableAlphaToCoverage();
    CHECK(other != blend);

    const nvrhi::BlendState::RenderTarget additive =
        nvrhi::utils::CreateAddBlendState(nvrhi::BlendFactor::One, nvrhi::BlendFactor::One);
    CHECK(additive.blendEnable);
    CHECK(additive.blendOp == nvrhi::BlendOp::Add);
    CHECK(additive.srcBlend == nvrhi::BlendFactor::One);
    CHECK(additive.destBlend == nvrhi::BlendFactor::One);
    CHECK(additive.srcBlendAlpha == nvrhi::BlendFactor::Zero);
    CHECK(additive.destBlendAlpha == nvrhi::BlendFactor::One);
}

NVRHI_TEST_CASE("binding layout items keep their type and size")
{
    const nvrhi::BindingLayoutItem texture = nvrhi::BindingLayoutItem::Texture_SRV(3);
    CHECK_EQ(texture.slot, 3u);
    CHECK(texture.type == nvrhi::ResourceType::Texture_SRV);
    CHECK_EQ(texture.size, uint16_t(1));
    CHECK_EQ(texture.getArraySize(), 1u);

    nvrhi::BindingLayoutItem array = texture;
    array.setSize(8);
    CHECK_EQ(array.getArraySize(), 8u);

    const nvrhi::BindingLayoutItem push = nvrhi::BindingLayoutItem::PushConstants(2, 64);
    CHECK_EQ(push.slot, 2u);
    CHECK(push.type == nvrhi::ResourceType::PushConstants);
    CHECK_EQ(push.size, uint16_t(64));
    CHECK_EQ(push.getArraySize(), 1u);

    CHECK(texture == nvrhi::BindingLayoutItem::Texture_SRV(3));
    CHECK(texture != nvrhi::BindingLayoutItem::Texture_SRV(4));
    CHECK_EQ(sizeof(nvrhi::BindingLayoutItem), size_t(8));
}

NVRHI_TEST_CASE("binding set items expose strongly typed factories")
{
    FakeTexture texture;
    texture.desc.setWidth(16).setHeight(16).setFormat(nvrhi::Format::RGBA8_UNORM);

    FakeBuffer buffer;
    buffer.desc.byteSize = 1024;

    FakeBuffer volatileBuffer;
    volatileBuffer.desc.byteSize = 256;
    volatileBuffer.desc.isVolatile = true;

    const nvrhi::BindingSetItem none = nvrhi::BindingSetItem::None(1);
    CHECK(none.type == nvrhi::ResourceType::None);
    CHECK_EQ(none.slot, 1u);
    CHECK(none.resourceHandle == nullptr);

    const nvrhi::BindingSetItem textureSrv =
        nvrhi::BindingSetItem::Texture_SRV(2, &texture);
    CHECK(textureSrv.type == nvrhi::ResourceType::Texture_SRV);
    CHECK(textureSrv.resourceHandle == &texture);
    CHECK(textureSrv.subresources == nvrhi::AllSubresources);
    CHECK(textureSrv.format == nvrhi::Format::UNKNOWN);
    CHECK(textureSrv.dimension == nvrhi::TextureDimension::Unknown);
    CHECK_EQ(textureSrv.overrideComponentMapping, uint16_t(0));

    const nvrhi::BindingSetItem textureUav =
        nvrhi::BindingSetItem::Texture_UAV(2, &texture);
    CHECK(textureUav.type == nvrhi::ResourceType::Texture_UAV);
    CHECK(textureUav.subresources == nvrhi::TextureSubresourceSet(0, 1, 0,
        nvrhi::TextureSubresourceSet::AllArraySlices));

    const nvrhi::BindingSetItem structured =
        nvrhi::BindingSetItem::StructuredBuffer_UAV(4, &buffer, nvrhi::Format::UNKNOWN,
            nvrhi::BufferRange(64, 128));
    CHECK(structured.type == nvrhi::ResourceType::StructuredBuffer_UAV);
    CHECK_EQ(structured.range.byteOffset, 64ull);
    CHECK_EQ(structured.range.byteSize, 128ull);

    const nvrhi::BindingSetItem constant = nvrhi::BindingSetItem::ConstantBuffer(0, &buffer);
    CHECK(constant.type == nvrhi::ResourceType::ConstantBuffer);

    const nvrhi::BindingSetItem volatileConstant =
        nvrhi::BindingSetItem::ConstantBuffer(0, &volatileBuffer);
    CHECK(volatileConstant.type == nvrhi::ResourceType::VolatileConstantBuffer);

    const nvrhi::BindingSetItem push = nvrhi::BindingSetItem::PushConstants(5, 32);
    CHECK(push.type == nvrhi::ResourceType::PushConstants);
    CHECK_EQ(push.range.byteSize, 32ull);

    nvrhi::BindingSetItem mapped = nvrhi::BindingSetItem::Texture_SRV(1, &texture)
        .setArrayElement(3)
        .setDimension(nvrhi::TextureDimension::Texture2DArray)
        .setFormat(nvrhi::Format::RGBA8_UNORM)
        .setOverrideComponentMapping(nvrhi::ComponentMapping{
            nvrhi::ComponentSwizzle::B, nvrhi::ComponentSwizzle::G,
            nvrhi::ComponentSwizzle::R, nvrhi::ComponentSwizzle::One });
    CHECK_EQ(mapped.arrayElement, 3u);
    CHECK((mapped.overrideComponentMapping & nvrhi::c_ComponentMappingExplicit) != 0);
    CHECK(nvrhi::resolveComponentMapping(mapped.overrideComponentMapping,
        nvrhi::ComponentMapping()).r == nvrhi::ComponentSwizzle::B);

    CHECK(mapped != textureSrv);
    CHECK(nvrhi::BindingSetItem::Sampler(0, nullptr) == nvrhi::BindingSetItem::Sampler(0, nullptr));
    CHECK_EQ(sizeof(nvrhi::BindingSetItem), size_t(40));
}

NVRHI_TEST_CASE("binding set descriptions compare and hash")
{
    FakeTexture texture;

    nvrhi::BindingSetDesc desc;
    desc.addItem(nvrhi::BindingSetItem::Texture_SRV(0, &texture));
    desc.addItem(nvrhi::BindingSetItem::Sampler(1, nullptr));

    nvrhi::BindingSetDesc equal;
    equal.addItem(nvrhi::BindingSetItem::Texture_SRV(0, &texture));
    equal.addItem(nvrhi::BindingSetItem::Sampler(1, nullptr));

    nvrhi::BindingSetDesc different;
    different.addItem(nvrhi::BindingSetItem::Texture_SRV(0, &texture));

    CHECK(desc == equal);
    CHECK(desc != different);
    CHECK_EQ(std::hash<nvrhi::BindingSetItem>()(desc.bindings[0]),
        std::hash<nvrhi::BindingSetItem>()(equal.bindings[0]));
    CHECK_EQ(std::hash<nvrhi::BindingSetDesc>()(desc),
        std::hash<nvrhi::BindingSetDesc>()(equal));
}

NVRHI_TEST_CASE("framebuffer descriptions derive pipeline compatibility info")
{
    FakeTexture color;
    color.desc.setWidth(64).setHeight(32).setFormat(nvrhi::Format::RGBA8_UNORM)
        .setSampleCount(1);

    FakeTexture depth;
    depth.desc.setWidth(64).setHeight(32).setFormat(nvrhi::Format::D32)
        .setSampleCount(1);

    const nvrhi::FramebufferAttachment emptyAttachment;
    CHECK(!emptyAttachment.valid());

    const nvrhi::FramebufferAttachment colorAttachment =
        nvrhi::FramebufferAttachment().setTexture(&color);
    CHECK(colorAttachment.valid());

    nvrhi::FramebufferDesc framebufferDesc;
    framebufferDesc.addColorAttachment(colorAttachment);
    framebufferDesc.setDepthAttachment(&depth);

    const nvrhi::FramebufferInfo info(framebufferDesc);
    CHECK_EQ(info.colorFormats.size(), size_t(1));
    CHECK(info.colorFormats[0] == nvrhi::Format::RGBA8_UNORM);
    CHECK(info.depthFormat == nvrhi::Format::D32);
    CHECK_EQ(info.sampleCount, 1u);
    CHECK_EQ(info.sampleQuality, 0u);

    nvrhi::FramebufferInfo equal;
    equal.addColorFormat(nvrhi::Format::RGBA8_UNORM);
    equal.setDepthFormat(nvrhi::Format::D32);
    CHECK(info == equal);

    nvrhi::FramebufferInfo different = equal;
    different.setSampleCount(4);
    CHECK(info != different);

    nvrhi::FramebufferDesc overridden;
    overridden.addColorAttachment(nvrhi::FramebufferAttachment()
        .setTexture(&color)
        .setFormat(nvrhi::Format::SRGBA8_UNORM));
    CHECK(nvrhi::FramebufferInfo(overridden).colorFormats[0] ==
        nvrhi::Format::SRGBA8_UNORM);

    nvrhi::FramebufferDesc sliced;
    sliced.addColorAttachment(nvrhi::FramebufferAttachment()
        .setTexture(&color)
        .setMipLevel(1));
    const nvrhi::FramebufferInfoEx extended(sliced);
    CHECK_EQ(extended.width, 32u);
    CHECK_EQ(extended.height, 16u);
    CHECK_EQ(extended.arraySize, 1u);

    const nvrhi::Viewport viewport = extended.getViewport(3.f, 7.f);
    CHECK_EQ(viewport.minX, 0.f);
    CHECK_EQ(viewport.maxX, 32.f);
    CHECK_EQ(viewport.minY, 0.f);
    CHECK_EQ(viewport.maxY, 16.f);
    CHECK_EQ(viewport.minZ, 3.f);
    CHECK_EQ(viewport.maxZ, 7.f);
}

NVRHI_TEST_CASE("utility descriptions are generated with their flags")
{
    const nvrhi::BufferDesc staticConstant =
        nvrhi::utils::CreateStaticConstantBufferDesc(64, "StaticConstants");
    CHECK_EQ(staticConstant.byteSize, 64ull);
    CHECK_EQ(staticConstant.debugName, std::string("StaticConstants"));
    CHECK(staticConstant.isConstantBuffer);
    CHECK(!staticConstant.isVolatile);

    const nvrhi::BufferDesc volatileConstant =
        nvrhi::utils::CreateVolatileConstantBufferDesc(256, "VolatileConstants", 4);
    CHECK_EQ(volatileConstant.byteSize, 256ull);
    CHECK(volatileConstant.isConstantBuffer);
    CHECK(volatileConstant.isVolatile);
    CHECK_EQ(volatileConstant.maxVersions, 4u);

    checkContains(nvrhi::utils::GenerateHeapDebugName(
        nvrhi::HeapDesc().setCapacity(1024).setType(nvrhi::HeapType::Upload)),
        "Upload Heap (1024 bytes)");
    checkContains(nvrhi::utils::GenerateHeapDebugName(
        nvrhi::HeapDesc().setCapacity(2048).setType(nvrhi::HeapType::Readback)),
        "Readback Heap (2048 bytes)");

    checkContains(nvrhi::utils::GenerateTextureDebugName(
        nvrhi::TextureDesc().setWidth(8).setHeight(4).setMipLevels(2)
            .setFormat(nvrhi::Format::RGBA8_UNORM)),
        "Unnamed Texture2D (RGBA8_UNORM, Width = 8, Height = 4, MipLevels = 2)");

    checkContains(nvrhi::utils::GenerateBufferDebugName(
        nvrhi::BufferDesc().setByteSize(1024).setCanHaveUAVs(true)),
        "CanHaveUAVs");
    checkContains(nvrhi::utils::GenerateBufferDebugName(
        nvrhi::BufferDesc().setByteSize(1024).setIsVertexBuffer(true)),
        "IsVertexBuffer");

    CHECK_EQ(nvrhi::utils::DebugNameToString(""), std::string("<UNNAMED>"));
    const std::string named = "Named";
    CHECK_EQ(std::string(nvrhi::utils::DebugNameToString(named)), std::string("Named"));
}

NVRHI_TEST_CASE("enum names are translated for diagnostics")
{
    CHECK_EQ(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::D3D11), std::string("D3D11"));
    CHECK_EQ(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::D3D12), std::string("D3D12"));
    CHECK_EQ(nvrhi::utils::GraphicsAPIToString(nvrhi::GraphicsAPI::VULKAN), std::string("Vulkan"));

    CHECK_EQ(nvrhi::utils::TextureDimensionToString(nvrhi::TextureDimension::Texture2D),
        std::string("Texture2D"));
    CHECK_EQ(nvrhi::utils::TextureDimensionToString(nvrhi::TextureDimension::Texture3D),
        std::string("Texture3D"));

    CHECK_EQ(nvrhi::utils::ShaderStageToString(nvrhi::ShaderType::Vertex),
        std::string("Vertex"));
    CHECK_EQ(nvrhi::utils::ShaderStageToString(nvrhi::ShaderType::RayGeneration),
        std::string("RayGeneration"));

    CHECK_EQ(nvrhi::utils::ResourceTypeToString(nvrhi::ResourceType::StructuredBuffer_UAV),
        std::string("StructuredBuffer_UAV"));
    CHECK_EQ(nvrhi::utils::ResourceTypeToString(nvrhi::ResourceType::PushConstants),
        std::string("PushConstants"));

    CHECK_EQ(nvrhi::utils::CommandQueueToString(nvrhi::CommandQueue::Graphics),
        std::string("Graphics"));
    CHECK_EQ(nvrhi::utils::CommandQueueToString(nvrhi::CommandQueue::Copy),
        std::string("Copy"));
}

NVRHI_TEST_CASE("BitSetAllocator recycles released indices")
{
    nvrhi::utils::BitSetAllocator allocator(3, false);
    CHECK_EQ(allocator.getCapacity(), size_t(3));
    CHECK_EQ(allocator.allocate(), 0);
    CHECK_EQ(allocator.allocate(), 1);
    CHECK_EQ(allocator.allocate(), 2);
    CHECK_EQ(allocator.allocate(), -1);

    allocator.release(1);
    CHECK_EQ(allocator.allocate(), 1);

    allocator.release(-1);
    allocator.release(42);

    nvrhi::utils::BitSetAllocator threaded(2, true);
    CHECK_EQ(threaded.allocate(), 0);
    CHECK_EQ(threaded.allocate(), 1);
    CHECK_EQ(threaded.allocate(), -1);
}

NVRHI_TEST_CASE("cooperative vector helpers report element sizes and strides")
{
    using nvrhi::coopvec::DataType;
    using nvrhi::coopvec::MatrixLayout;

    const std::pair<DataType, size_t> sizes[] = {
        { DataType::UInt8, 1 }, { DataType::SInt8, 1 },
        { DataType::UInt8Packed, 1 }, { DataType::SInt8Packed, 1 },
        { DataType::UInt16, 2 }, { DataType::SInt16, 2 },
        { DataType::UInt32, 4 }, { DataType::SInt32, 4 },
        { DataType::UInt64, 8 }, { DataType::SInt64, 8 },
        { DataType::FloatE4M3, 1 }, { DataType::FloatE5M2, 1 },
        { DataType::Float16, 2 }, { DataType::BFloat16, 2 },
        { DataType::Float32, 4 }, { DataType::Float64, 8 },
    };

    for (const auto& entry : sizes)
        CHECK_EQ(nvrhi::coopvec::getDataTypeSize(entry.first), entry.second);

    CHECK_EQ(nvrhi::coopvec::getOptimalMatrixStride(DataType::Float16, MatrixLayout::RowMajor,
        4, 8), size_t(16));
    CHECK_EQ(nvrhi::coopvec::getOptimalMatrixStride(DataType::Float16, MatrixLayout::ColumnMajor,
        4, 8), size_t(8));
    CHECK_EQ(nvrhi::coopvec::getOptimalMatrixStride(DataType::Float32,
        MatrixLayout::InferencingOptimal, 4, 8), size_t(0));
    CHECK_EQ(nvrhi::coopvec::getOptimalMatrixStride(DataType::Float32,
        MatrixLayout::TrainingOptimal, 4, 8), size_t(0));
}

NVRHI_TEST_CASE("enum flag operators combine and mask values")
{
    const nvrhi::FormatSupport combined =
        nvrhi::FormatSupport::Buffer | nvrhi::FormatSupport::Texture;
    CHECK((combined & nvrhi::FormatSupport::Buffer) == nvrhi::FormatSupport::Buffer);
    CHECK_EQ(uint32_t(combined), uint32_t(nvrhi::FormatSupport::Buffer) |
        uint32_t(nvrhi::FormatSupport::Texture));
    CHECK(!nvrhi::FormatSupport::None);
    CHECK(!!nvrhi::FormatSupport::Buffer);
    CHECK(((~nvrhi::FormatSupport::None) & nvrhi::FormatSupport::RenderTarget) ==
        nvrhi::FormatSupport::RenderTarget);

    const nvrhi::ShaderType graphics = nvrhi::ShaderType::AllGraphics;
    CHECK((graphics & nvrhi::ShaderType::Vertex) != nvrhi::ShaderType::None);
    CHECK((graphics & nvrhi::ShaderType::Pixel) != nvrhi::ShaderType::None);
    CHECK((graphics & nvrhi::ShaderType::Compute) == nvrhi::ShaderType::None);

    const nvrhi::ResourceStates shaderResource = nvrhi::ResourceStates::ShaderResource;
    CHECK((shaderResource & nvrhi::ResourceStates::PixelShaderResource) !=
        nvrhi::ResourceStates::Unknown);
    CHECK((shaderResource & nvrhi::ResourceStates::NonPixelShaderResource) !=
        nvrhi::ResourceStates::Unknown);
}

NVRHI_TEST_CASE("ray tracing descriptions keep their CPU-side layout")
{
    CHECK_EQ(sizeof(nvrhi::rt::InstanceDesc), size_t(64));

    nvrhi::rt::InstanceDesc instance;
    CHECK_EQ(instance.instanceID, 0u);
    CHECK_EQ(instance.instanceMask, 0u);
    CHECK(instance.bottomLevelAS == nullptr);
    CHECK_NEAR(instance.transform[0], 1.f, 1e-6f);
    CHECK_NEAR(instance.transform[5], 1.f, 1e-6f);
    CHECK_NEAR(instance.transform[10], 1.f, 1e-6f);

    instance.setInstanceID(7).setInstanceMask(0xFF)
        .setInstanceContributionToHitGroupIndex(3)
        .setFlags(nvrhi::rt::InstanceFlags::ForceOpaque);
    CHECK_EQ(instance.instanceID, 7u);
    CHECK_EQ(instance.instanceMask, 0xFFu);
    CHECK_EQ(instance.instanceContributionToHitGroupIndex, 3u);
    CHECK((instance.flags & nvrhi::rt::InstanceFlags::ForceOpaque) !=
        nvrhi::rt::InstanceFlags::None);

    nvrhi::rt::AccelStructDesc topLevel;
    topLevel.setTopLevelMaxInstances(16);
    CHECK(topLevel.isTopLevel);
    CHECK_EQ(topLevel.topLevelMaxInstances, size_t(16));

    topLevel.addBottomLevelGeometry(nvrhi::rt::GeometryDesc());
    CHECK(!topLevel.isTopLevel);
    CHECK_EQ(topLevel.bottomLevelGeometries.size(), size_t(1));

    nvrhi::rt::GeometryTriangles triangles;
    triangles.setVertexBuffer(nullptr).setVertexFormat(nvrhi::Format::RGB32_FLOAT)
        .setVertexCount(3).setVertexStride(12);
    CHECK_EQ(triangles.vertexCount, 3u);
    CHECK_EQ(triangles.vertexStride, 12u);

    nvrhi::rt::GeometryDesc geometry;
    geometry.setTriangles(triangles);
    CHECK(geometry.geometryType == nvrhi::rt::GeometryType::Triangles);
    CHECK(!geometry.useTransform);

    const nvrhi::rt::AffineTransform translation = {
        1.f, 0.f, 0.f, 4.f,
        0.f, 1.f, 0.f, 5.f,
        0.f, 0.f, 1.f, 6.f };
    geometry.setTransform(translation);
    CHECK(geometry.useTransform);
    CHECK_NEAR(geometry.transform[3], 4.f, 1e-6f);
    CHECK_NEAR(geometry.transform[7], 5.f, 1e-6f);
    CHECK_NEAR(geometry.transform[11], 6.f, 1e-6f);
}

NVRHI_TEST_CASE("shader specializations carry their value type")
{
    const nvrhi::ShaderSpecialization uintValue =
        nvrhi::ShaderSpecialization::UInt32(3, 7);
    CHECK_EQ(uintValue.constantID, 3u);
    CHECK_EQ(uintValue.value.u, 7u);

    const nvrhi::ShaderSpecialization intValue =
        nvrhi::ShaderSpecialization::Int32(4, -9);
    CHECK_EQ(intValue.constantID, 4u);
    CHECK_EQ(intValue.value.i, -9);

    const nvrhi::ShaderSpecialization floatValue =
        nvrhi::ShaderSpecialization::Float(5, 0.5f);
    CHECK_EQ(floatValue.constantID, 5u);
    CHECK_EQ(floatValue.value.f, 0.5f);
}

NVRHI_TEST_CASE("resource binding keys compare through their base ranges")
{
    const nvrhi::TextureSubresourceSet subresources(1, 2, 0, 3);
    const nvrhi::TextureBindingKey textureKey(subresources, nvrhi::Format::RGBA8_UNORM);
    const nvrhi::TextureBindingKey equalKey(subresources, nvrhi::Format::RGBA8_UNORM);
    const nvrhi::TextureBindingKey otherFormat(subresources, nvrhi::Format::BGRA8_UNORM);
    const nvrhi::TextureBindingKey readOnlyKey(subresources, nvrhi::Format::RGBA8_UNORM, true);
    const nvrhi::TextureBindingKey swizzledKey(subresources, nvrhi::Format::RGBA8_UNORM,
        false, 1);

    CHECK(textureKey == equalKey);
    CHECK(!(textureKey == otherFormat));
    CHECK(!(textureKey == readOnlyKey));
    CHECK(!(textureKey == swizzledKey));

    std::unordered_map<nvrhi::TextureBindingKey, int> textureMap;
    textureMap[textureKey] = 5;
    CHECK_EQ(textureMap[equalKey], 5);

    const nvrhi::BufferRange range(64, 128);
    const nvrhi::BufferBindingKey bufferKey(range, nvrhi::Format::R32_UINT,
        nvrhi::ResourceType::StructuredBuffer_SRV);
    const nvrhi::BufferBindingKey equalBufferKey(range, nvrhi::Format::R32_UINT,
        nvrhi::ResourceType::StructuredBuffer_SRV);

    CHECK(bufferKey == equalBufferKey);
    CHECK(!(bufferKey == nvrhi::BufferBindingKey(range, nvrhi::Format::R32_UINT,
        nvrhi::ResourceType::RawBuffer_SRV)));
    CHECK(!(bufferKey == nvrhi::BufferBindingKey(nvrhi::BufferRange(64, 64),
        nvrhi::Format::R32_UINT, nvrhi::ResourceType::StructuredBuffer_SRV)));
}

NVRHI_TEST_CASE("header version checks the implementation version")
{
    CHECK(nvrhi::verifyHeaderVersion());
    CHECK(nvrhi::verifyHeaderVersion(nvrhi::c_HeaderVersion));
    CHECK(!nvrhi::verifyHeaderVersion(nvrhi::c_HeaderVersion + 1));
}

NVRHI_TEST_CASE("IResource supplies safe default queries and reference counting")
{
    FakeResource<nvrhi::IResource> resource;
    CHECK_EQ(resource.GetRefCount(), 1ul);
    CHECK_EQ(resource.AddRef(), 2ul);
    CHECK_EQ(resource.GetRefCount(), 2ul);
    CHECK_EQ(resource.Release(), 1ul);

    nvrhi::MemoryRequirements requirements = { 11, 22 };
    CHECK(!resource.queryMemoryRequirements(requirements));
    CHECK_EQ(requirements.size, 11ull);
    CHECK_EQ(requirements.alignment, 22ull);

    const nvrhi::Object native =
        resource.getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
    CHECK(native.pointer == nullptr);
}

NVRHI_TEST_CASE("hash specializations agree for equal values")
{
    const nvrhi::TextureSubresourceSet subresources(0, 2, 0, 4);
    CHECK_EQ(std::hash<nvrhi::TextureSubresourceSet>()(subresources),
        std::hash<nvrhi::TextureSubresourceSet>()(nvrhi::TextureSubresourceSet(0, 2, 0, 4)));

    const nvrhi::BufferRange range(16, 32);
    CHECK_EQ(std::hash<nvrhi::BufferRange>()(range),
        std::hash<nvrhi::BufferRange>()(nvrhi::BufferRange(16, 32)));

    nvrhi::BlendState blend;
    nvrhi::BlendState equalBlend;
    CHECK_EQ(std::hash<nvrhi::BlendState>()(blend),
        std::hash<nvrhi::BlendState>()(equalBlend));

    nvrhi::FramebufferInfo info;
    info.addColorFormat(nvrhi::Format::RGBA8_UNORM);
    nvrhi::FramebufferInfo equalInfo;
    equalInfo.addColorFormat(nvrhi::Format::RGBA8_UNORM);
    CHECK_EQ(std::hash<nvrhi::FramebufferInfo>()(info),
        std::hash<nvrhi::FramebufferInfo>()(equalInfo));

    size_t seedA = 0;
    size_t seedB = 0;
    nvrhi::hash_combine(seedA, 17u);
    nvrhi::hash_combine(seedB, 17u);
    CHECK_EQ(seedA, seedB);
}

NVRHI_TEST_CASE("draw and dispatch arguments expose their defaults")
{
    const nvrhi::DrawArguments draw;
    CHECK_EQ(draw.vertexCount, 0u);
    CHECK_EQ(draw.instanceCount, 1u);

    const nvrhi::DrawIndirectArguments indirect =
        nvrhi::DrawIndirectArguments().setVertexCount(3).setInstanceCount(2);
    CHECK_EQ(indirect.vertexCount, 3u);
    CHECK_EQ(indirect.instanceCount, 2u);

    const nvrhi::DrawIndexedIndirectArguments indexed =
        nvrhi::DrawIndexedIndirectArguments().setIndexCount(6).setBaseVertexLocation(-2);
    CHECK_EQ(indexed.indexCount, 6u);
    CHECK_EQ(indexed.baseVertexLocation, -2);

    const nvrhi::DispatchIndirectArguments dispatch =
        nvrhi::DispatchIndirectArguments().setGroups3D(2, 3, 4);
    CHECK_EQ(dispatch.groupsX, 2u);
    CHECK_EQ(dispatch.groupsY, 3u);
    CHECK_EQ(dispatch.groupsZ, 4u);

    nvrhi::ViewportState viewportState;
    viewportState.addViewport(nvrhi::Viewport(8.f, 8.f));
    viewportState.addViewportAndScissorRect(nvrhi::Viewport(4.f, 4.f));
    CHECK_EQ(viewportState.viewports.size(), size_t(2));
    CHECK_EQ(viewportState.scissorRects.size(), size_t(1));

    CHECK(nvrhi::Rect(0, 4, 0, 4) == nvrhi::Rect(4, 4));
}
