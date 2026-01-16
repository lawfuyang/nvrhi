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

#include "vulkan-backend.h"
#include <nvrhi/common/misc.h>

namespace nvrhi::vulkan
{

    EventQueryHandle Device::createEventQuery(void)
    {
        EventQuery *query = new EventQuery();
        return EventQueryHandle::Create(query);
    }

    void Device::setEventQuery(IEventQuery* _query, CommandQueue queue)
    {
        EventQuery* query = checked_cast<EventQuery*>(_query);

        assert(query->commandListID == 0);

        query->queue = queue;
        query->commandListID = m_Queues[uint32_t(queue)]->getLastSubmittedID();
    }

    bool Device::pollEventQuery(IEventQuery* _query)
    {
        EventQuery* query = checked_cast<EventQuery*>(_query);
        
        auto& queue = *m_Queues[uint32_t(query->queue)];

        return queue.pollCommandList(query->commandListID);
    }

    void Device::waitEventQuery(IEventQuery* _query)
    {
        EventQuery* query = checked_cast<EventQuery*>(_query);

        if (query->commandListID == 0)
            return;

        auto& queue = *m_Queues[uint32_t(query->queue)];

        bool success = queue.waitCommandList(query->commandListID, ~0ull);
        assert(success);
        (void)success;
    }

    void Device::resetEventQuery(IEventQuery* _query)
    {
        EventQuery* query = checked_cast<EventQuery*>(_query);

        query->commandListID = 0;
    }


    TimerQueryHandle Device::createTimerQuery(void)
    {
        if (!m_TimerQueryPool)
        {
            std::lock_guard lockGuard(m_Mutex);

            if (!m_TimerQueryPool)
            {
                // set up the timer query pool on first use
                auto poolInfo = vk::QueryPoolCreateInfo()
                    .setQueryType(vk::QueryType::eTimestamp)
                    .setQueryCount(uint32_t(m_TimerQueryAllocator.getCapacity()) * 2); // use 2 Vulkan queries per 1 TimerQuery

                const vk::Result res = m_Context.device.createQueryPool(&poolInfo, m_Context.allocationCallbacks, &m_TimerQueryPool);
                CHECK_VK_FAIL(res)
            }
        }

        int queryIndex = m_TimerQueryAllocator.allocate();

        if (queryIndex < 0)
        {
            m_Context.error("Insufficient query pool space, increase Device::numTimerQueries");
            return nullptr;
        }

        TimerQuery* query = new TimerQuery(m_TimerQueryAllocator);
        query->beginQueryIndex = queryIndex * 2;
        query->endQueryIndex = queryIndex * 2 + 1;

        return TimerQueryHandle::Create(query);
    }

    TimerQuery::~TimerQuery()
    {
        m_QueryAllocator.release(beginQueryIndex / 2);
        beginQueryIndex = -1;
        endQueryIndex = -1;
    }

    // [rlaw] BEGIN: Pipeline Query support
    PipelineStatisticsQuery::~PipelineStatisticsQuery()
    {
        m_QueryAllocator.release(queryIndex);
        queryIndex = -1;
    }
    // [rlaw] END: Pipeline Query support

    void CommandList::beginTimerQuery(ITimerQuery* _query)
    {
        endRenderPass();

        TimerQuery* query = checked_cast<TimerQuery*>(_query);

        assert(query->beginQueryIndex >= 0);
        assert(!query->started);
        assert(m_CurrentCmdBuf);

        query->resolved = false;

        m_CurrentCmdBuf->cmdBuf.resetQueryPool(m_Device->getTimerQueryPool(), query->beginQueryIndex, 2);
        m_CurrentCmdBuf->cmdBuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_Device->getTimerQueryPool(), query->beginQueryIndex);
    }

    void CommandList::endTimerQuery(ITimerQuery* _query)
    {
        endRenderPass();

        TimerQuery* query = checked_cast<TimerQuery*>(_query);

        assert(query->endQueryIndex >= 0);
        assert(!query->started);
        assert(!query->resolved);

        assert(m_CurrentCmdBuf);

        m_CurrentCmdBuf->cmdBuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_Device->getTimerQueryPool(), query->endQueryIndex);
        query->started = true;
    }

    bool Device::pollTimerQuery(ITimerQuery* _query)
    {
        TimerQuery* query = checked_cast<TimerQuery*>(_query);

        if (!query->started)
        {
            return false;
        }

        if (query->resolved)
        {
            return true;
        }

        uint32_t timestamps[2] = { 0, 0 };

        vk::Result res;
        res = m_Context.device.getQueryPoolResults(m_TimerQueryPool,
                                                 query->beginQueryIndex, 2,
                                                 sizeof(timestamps), timestamps,
                                                 sizeof(timestamps[0]), vk::QueryResultFlags());
        assert(res == vk::Result::eSuccess || res == vk::Result::eNotReady || res == vk::Result::eErrorDeviceLost);

        if (res == vk::Result::eNotReady || res == vk::Result::eErrorDeviceLost)
        {
            return false;
        }

        const auto timestampPeriod = m_Context.physicalDeviceProperties.limits.timestampPeriod; // in nanoseconds
        const float scale = 1e-9f * timestampPeriod;

        query->time = float(timestamps[1] - timestamps[0]) * scale;
        query->resolved = true;
        return true;
    }

    float Device::getTimerQueryTime(ITimerQuery* _query)
    {
        TimerQuery* query = checked_cast<TimerQuery*>(_query);

        if (!query->started)
            return 0.f;

        if (!query->resolved)
        {
            while(!pollTimerQuery(query))
                ;
        }

        query->started = false;

        assert(query->resolved);
        return query->time;
    }

    void Device::resetTimerQuery(ITimerQuery* _query)
    {
        TimerQuery* query = checked_cast<TimerQuery*>(_query);

        query->started = false;
        query->resolved = false;
        query->time = 0.f;
    }

    // [rlaw] BEGIN: Pipeline Query support
    PipelineStatisticsQueryHandle Device::createPipelineStatisticsQuery()
    {
        if (!m_PipelineStatisticsQueryPool)
        {
            std::lock_guard lockGuard(m_Mutex);

            if (!m_PipelineStatisticsQueryPool)
            {
                // set up the pipeline statistics query pool on first use
                vk::QueryPipelineStatisticFlags flags = 
                    vk::QueryPipelineStatisticFlagBits::eInputAssemblyVertices |
                    vk::QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives |
                    vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations |
                    vk::QueryPipelineStatisticFlagBits::eGeometryShaderInvocations |
                    vk::QueryPipelineStatisticFlagBits::eGeometryShaderPrimitives |
                    vk::QueryPipelineStatisticFlagBits::eClippingInvocations |
                    vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
                    vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations |
                    vk::QueryPipelineStatisticFlagBits::eTessellationControlShaderPatches |
                    vk::QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations |
                    vk::QueryPipelineStatisticFlagBits::eComputeShaderInvocations;

                if (m_Context.extensions.NV_mesh_shader)
                {
                    flags |= vk::QueryPipelineStatisticFlagBits::eTaskShaderInvocationsEXT |
                             vk::QueryPipelineStatisticFlagBits::eMeshShaderInvocationsEXT;
                }

                auto poolInfo = vk::QueryPoolCreateInfo()
                    .setQueryType(vk::QueryType::ePipelineStatistics)
                    .setQueryCount(uint32_t(m_PipelineStatisticsQueryAllocator.getCapacity()))
                    .setPipelineStatistics(flags);

                const vk::Result res = m_Context.device.createQueryPool(&poolInfo, m_Context.allocationCallbacks, &m_PipelineStatisticsQueryPool);
                CHECK_VK_FAIL(res)
            }
        }

        int queryIndex = m_PipelineStatisticsQueryAllocator.allocate();

        if (queryIndex < 0)
        {
            m_Context.error("Insufficient pipeline statistics query pool space");
            return nullptr;
        }

        PipelineStatisticsQuery* query = new PipelineStatisticsQuery(m_PipelineStatisticsQueryAllocator);
        query->queryIndex = queryIndex;

        return PipelineStatisticsQueryHandle::Create(query);
    }
    
    PipelineStatistics Device::getPipelineStatistics(IPipelineStatisticsQuery* _query)
    {
        PipelineStatisticsQuery* query = checked_cast<PipelineStatisticsQuery*>(_query);

        if (!query->resolved)
        {
            constexpr uint32_t MaxPipelineStatistics = 13; // Maximum number of statistics we can query
            uint64_t data[MaxPipelineStatistics]{};

            const uint32_t numStats = m_Context.extensions.NV_mesh_shader ? 13 : 11;

            const vk::Result res = m_Context.device.getQueryPoolResults(
                m_PipelineStatisticsQueryPool,
                query->queryIndex,
                1,
                numStats * sizeof(uint64_t),
                data,
                sizeof(uint64_t),
                vk::QueryResultFlagBits::e64);

            if (res == vk::Result::eSuccess)
            {
                query->resolved = true;
                query->statistics.IAVertices = data[0];  // INPUT_ASSEMBLY_VERTICES
                query->statistics.IAPrimitives = data[1]; // INPUT_ASSEMBLY_PRIMITIVES
                query->statistics.VSInvocations = data[2]; // VERTEX_SHADER_INVOCATIONS
                query->statistics.GSInvocations = data[3]; // GEOMETRY_SHADER_INVOCATIONS
                query->statistics.GSPrimitives = data[4]; // GEOMETRY_SHADER_PRIMITIVES
                query->statistics.CInvocations = data[5]; // CLIPPING_INVOCATIONS
                query->statistics.CPrimitives = data[6]; // CLIPPING_PRIMITIVES
                query->statistics.PSInvocations = data[7]; // FRAGMENT_SHADER_INVOCATIONS
                query->statistics.HSInvocations = data[8]; // TESSELLATION_CONTROL_SHADER_PATCHES
                query->statistics.DSInvocations = data[9]; // TESSELLATION_EVALUATION_SHADER_INVOCATIONS
                query->statistics.CSInvocations = data[10]; // COMPUTE_SHADER_INVOCATIONS
                if (m_Context.extensions.NV_mesh_shader)
                {
                    query->statistics.ASInvocations = data[11]; // TASK_SHADER_INVOCATIONS_EXT
                    query->statistics.MSInvocations = data[12]; // MESH_SHADER_INVOCATIONS_EXT
                }
                // MSPrimitives is not available in Vulkan
            }
        }

        return query->statistics;
    }
    
    bool Device::pollPipelineStatisticsQuery(IPipelineStatisticsQuery* _query)
    {
        PipelineStatisticsQuery* query = checked_cast<PipelineStatisticsQuery*>(_query);

        if (!query->started)
            return false;

        // For Vulkan, we can check if the query is available
        constexpr uint32_t MaxPipelineStatistics = 13; // Maximum number of statistics we can query
        uint64_t data[MaxPipelineStatistics]{};

        const vk::Result res = m_Context.device.getQueryPoolResults(
            m_PipelineStatisticsQueryPool,
            query->queryIndex,
            1,
            sizeof(data),
            data,
            0,
            vk::QueryResultFlagBits::eWait);

        return res == vk::Result::eSuccess;
    }

    void Device::resetPipelineStatisticsQuery(IPipelineStatisticsQuery* _query)
    {
        PipelineStatisticsQuery* query = checked_cast<PipelineStatisticsQuery*>(_query);

        query->started = false;
        query->resolved = false;
        memset(&query->statistics, 0, sizeof(PipelineStatistics));
    }
    
    void CommandList::beginPipelineStatisticsQuery(IPipelineStatisticsQuery* _query)
    {
        PipelineStatisticsQuery* query = checked_cast<PipelineStatisticsQuery*>(_query);

        assert(query->queryIndex >= 0);
        assert(!query->started);
        assert(m_CurrentCmdBuf);

        query->resolved = false;

        m_CurrentCmdBuf->cmdBuf.resetQueryPool(m_Device->getPipelineStatisticsQueryPool(), query->queryIndex, 1);
        m_CurrentCmdBuf->cmdBuf.beginQuery(m_Device->getPipelineStatisticsQueryPool(), query->queryIndex, vk::QueryControlFlags());
    }

    void CommandList::endPipelineStatisticsQuery(IPipelineStatisticsQuery* _query)
    {
        PipelineStatisticsQuery* query = checked_cast<PipelineStatisticsQuery*>(_query);

        assert(query->queryIndex >= 0);
        assert(!query->started);
        assert(!query->resolved);
        assert(m_CurrentCmdBuf);

        m_CurrentCmdBuf->cmdBuf.endQuery(m_Device->getPipelineStatisticsQueryPool(), query->queryIndex);
        query->started = true;
    }
    // [rlaw] END: Pipeline Query support

    void CommandList::beginMarker(const char* name)
    {
        if (m_Context.extensions.EXT_debug_utils)
        {
            assert(m_CurrentCmdBuf);

            auto label = vk::DebugUtilsLabelEXT()
                            .setPLabelName(name);
            m_CurrentCmdBuf->cmdBuf.beginDebugUtilsLabelEXT(&label);
        }
        else if (m_Context.extensions.EXT_debug_marker)
        {
            assert(m_CurrentCmdBuf);

            auto markerInfo = vk::DebugMarkerMarkerInfoEXT()
                                .setPMarkerName(name);
            m_CurrentCmdBuf->cmdBuf.debugMarkerBeginEXT(&markerInfo);
        }
        
#if NVRHI_WITH_AFTERMATH
        if (m_Device->isAftermathEnabled())
        {
            const size_t aftermathMarker = m_AftermathTracker.pushEvent(name);
            m_CurrentCmdBuf->cmdBuf.setCheckpointNV((const void*)aftermathMarker);
        }
#endif
    }

    void CommandList::endMarker()
    {
        if (m_Context.extensions.EXT_debug_utils)
        {
            assert(m_CurrentCmdBuf);

            m_CurrentCmdBuf->cmdBuf.endDebugUtilsLabelEXT();
        }
        else if (m_Context.extensions.EXT_debug_marker)
        {
            assert(m_CurrentCmdBuf);

            m_CurrentCmdBuf->cmdBuf.debugMarkerEndEXT();
        }
        
#if NVRHI_WITH_AFTERMATH
        m_AftermathTracker.popEvent();
#endif
    }

} // namespace nvrhi::vulkan